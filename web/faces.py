"""Finding the same person across a library, on the machine, for nothing.

No API is involved here and nothing leaves the server.  Two small ONNX models
run under OpenCV: YuNet finds faces, SFace turns each one into 128 numbers.
Measured on a 1600 pixel photograph with twenty-four faces in it: 35 ms.  Beside
the seconds the codec already spends per photograph, that is free.

**It runs at upload, not later.**  Unlike captioning, this needs the actual
pixels at a decent size, and the one moment the full image is already in memory
is while it is being encoded.  Decoding it again afterwards would cost seconds
of neural network per photograph to recover something we were holding for free.

**Small faces are thrown away, and that is the whole trick.**  Measured on a
crowd shot: across every pair of *different* people, cosine similarity averaged
0.098, but seven pairs out of 190 crossed the 0.363 matching threshold and would
have been merged into one person.  Every one of those was a background
spectator twelve pixels across, where the embedding is noise.  Filtering to
faces at least 40 pixels on their short edge removed all of them.  It also
happens to be what you want anyway: strangers in the back of a crowd do not
belong in a list of people you know.

**Biometric data, so it is off until asked.**  These vectors identify people who
never agreed to anything, most of them not the person who owns the library.
The switch lives per library, defaults off, and turning it off deletes
everything derived here rather than merely hiding it.
"""

import io
import os
from pathlib import Path

# Where the two models live. Not in the repo: 37 MB of weights does not belong in
# git, and the feature degrades to "unavailable" without them rather than
# breaking the app. `scripts/get_face_models.py` fetches them.
MODEL_DIR = Path(os.environ.get("JSONCAM_FACE_MODELS", "checkpoints/faces"))
DETECTOR = MODEL_DIR / "yunet.onnx"
EMBEDDER = MODEL_DIR / "sface.onnx"

# How sure YuNet has to be that this is a face at all.
CONFIDENCE = float(os.environ.get("JSONCAM_FACE_CONFIDENCE", "0.8"))

# The short edge of the face box, in pixels. Below this the embedding stops
# describing a person and starts describing noise; see the module docstring for
# the measurement this number comes from.
MIN_EDGE = int(os.environ.get("JSONCAM_FACE_MIN_EDGE", "40"))

# Cosine similarity above which two faces are treated as the same person.
#
# Not OpenCV's published 0.363, and the difference matters. That figure is
# calibrated for *verification*: given two photographs, are these the same
# person, where a mistake costs one wrong answer. This is *clustering*, where a
# mistake chains: one borderline face joining the wrong pile drags a whole
# person's photographs in with it, because the centroid then sits between two
# people and attracts both.
#
# Measured on a photograph of six visibly different people, faces 61-116px:
# every genuinely different pair scored at or below 0.477, with three pairs
# between 0.36 and 0.48 that 0.363 merged into one person. The same face
# re-embedded after blurring scored 0.977. So on this data the two populations
# are separated by a wide gap, and 0.363 sits inside the wrong one.
#
# 0.5 sits in that gap. It will still split one person into two piles when
# lighting or angle changes a lot between photographs, and that is the
# deliberate direction to fail in: splitting is a five second fix with the merge
# button, while a wrong merge quietly loses the distinction between two people.
#
# This wants calibrating against a real library. It is an env var for that
# reason.
MATCH = float(os.environ.get("JSONCAM_FACE_MATCH", "0.5"))

# How much better the best match must be than the runner-up before a face is
# assigned rather than made into a new person.
#
# This is the other half of the chaining defence. A face that looks almost
# equally like two existing people is exactly the face that bridges them, and
# the honest answer for it is "I do not know", which here means a new pile that
# somebody can merge deliberately.
MARGIN = float(os.environ.get("JSONCAM_FACE_MARGIN", "0.06"))

# A face thumbnail for the people picker. 96px is enough to recognise somebody
# and costs about 3 KB.
CROP_SIDE = 96

_models = {}


def available():
    """Whether faces can be found at all, without loading anything."""
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return DETECTOR.is_file() and EMBEDDER.is_file()


def _load():
    """Both models, loaded once and reused. They are not cheap to construct."""
    if "det" not in _models:
        import cv2

        _models["det"] = cv2.FaceDetectorYN.create(
            str(DETECTOR), "", (320, 320), CONFIDENCE, 0.3, 5000)
        _models["rec"] = cv2.FaceRecognizerSF.create(str(EMBEDDER), "")
    return _models["det"], _models["rec"]


def find(image):
    """Every usable face in a PIL image.

    Returns a list of dicts with a normalised bounding box, a unit-length
    128-float embedding, a small JPEG crop and the detector's confidence.
    Returns an empty list on any failure: a photograph must still upload when
    the face models are missing or a frame confuses the detector.
    """
    if not available():
        return []
    try:
        import cv2
        import numpy as np

        detector, recogniser = _load()
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        detector.setInputSize((width, height))
        _, found = detector.detect(bgr)
        if found is None:
            return []

        out = []
        for row in found:
            x, y, box_w, box_h = (float(v) for v in row[:4])
            score = float(row[-1])
            if min(box_w, box_h) < MIN_EDGE:
                continue                      # too small to identify anybody

            aligned = recogniser.alignCrop(bgr, row)
            vector = recogniser.feature(aligned)[0].astype("float32")
            norm = float(np.linalg.norm(vector))
            if not norm:
                continue
            vector = vector / norm            # unit length, so a dot product is cosine

            crop = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            from PIL import Image

            thumb = Image.fromarray(crop).resize((CROP_SIDE, CROP_SIDE), Image.LANCZOS)
            buffer = io.BytesIO()
            thumb.save(buffer, "JPEG", quality=78, optimize=True)

            out.append({
                # Normalised, so the box still means something after a resize.
                "bbox": [round(x / width, 5), round(y / height, 5),
                         round(box_w / width, 5), round(box_h / height, 5)],
                "embedding": vector,
                "crop": buffer.getvalue(),
                "score": round(score, 4),
                "edge": int(min(box_w, box_h)),
            })
        return out
    except Exception:
        return []


def to_blob(vector):
    """A unit embedding as bytes, for a SQLite column."""
    return vector.astype("float32").tobytes() if hasattr(vector, "astype") else bytes(vector)


def from_blob(blob):
    import numpy as np

    return np.frombuffer(blob, dtype="float32")


def similarity(a, b):
    """Cosine similarity between two unit vectors, which is just a dot product."""
    import numpy as np

    return float(np.dot(a, b))


def merged_centroid(centroid, count, vector):
    """The running mean of a person's faces, kept unit length.

    Stored rather than recomputed so that matching a new face is a comparison
    against one vector per person, not against every face ever filed. A library
    with four hundred photographs of six people costs six dot products per new
    face, which is what makes this cheap enough to do inline at upload.
    """
    import numpy as np

    blended = (np.asarray(centroid, dtype="float32") * count + vector) / (count + 1)
    norm = float(np.linalg.norm(blended))
    return blended / norm if norm else blended
