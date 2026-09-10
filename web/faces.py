"""Finding the same person across a library, on the machine, for nothing.

No API is involved here and nothing leaves the server.  Two small ONNX models
run under OpenCV: YuNet finds faces, SFace turns each one into 128 numbers.
Measured on a 1600 pixel photograph with twenty-four faces in it: 35 ms.  Beside
the seconds the codec already spends per photograph, that is free.

**It runs at upload, not later.**  Unlike captioning, this needs the actual
pixels at a decent size, and the one moment the full image is already in memory
is while it is being encoded.  Decoding it again afterwards would cost seconds
of neural network per photograph to recover something we were holding for free.

**Faces it cannot see properly are thrown away rather than guessed at.**  Four
gates, and every one of them exists because of a face that broke clustering:
too small, eyes too close together to be face-on, nose too far off the eye
midpoint (a profile), or too blurred.  Measured on one photograph, the largest
face in the frame was in hard profile with the eyes 19 pixels apart, and its
embedding described a silhouette; size alone had let it through.

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
# the measurement this number comes from. Raised from 40 after false merges
# kept coming from the smallest faces that cleared the old bar.
MIN_EDGE = int(os.environ.get("JSONCAM_FACE_MIN_EDGE", "60"))

# How far the nose may sit from the midpoint of the eyes, as a fraction of the
# distance between them. Near zero is face-on; large means the head is turned.
#
# This is the single most useful thing YuNet's landmarks buy. Measured on one
# photograph: three faces at 0.08, 0.13 and 0.14, and a fourth at 1.55 which was
# the *largest* face in the frame but in hard profile, with the eyes only 19
# pixels apart. A profile embedding describes a silhouette rather than a person,
# and that face is why size alone was not enough of a gate.
MAX_YAW = float(os.environ.get("JSONCAM_FACE_MAX_YAW", "0.45"))

# The eyes must be at least this fraction of the face box apart. Catches the
# same problem from the other side, plus bad landmark fits.
MIN_EYE_RATIO = float(os.environ.get("JSONCAM_FACE_MIN_EYE_RATIO", "0.22"))

# Variance of the Laplacian over the crop: a blur measure. A motion-blurred face
# embeds as an average of everybody. Measured usable faces scored 400 to 1300.
MIN_SHARPNESS = float(os.environ.get("JSONCAM_FACE_MIN_SHARPNESS", "60"))

# Cosine similarity above which two faces are treated as the same person.
#
# Not OpenCV's published 0.363, and the difference matters. That figure is
# calibrated for *verification*: given two photographs, are these the same
# person, where a mistake costs one wrong answer. This is *clustering*, where a
# mistake chains: one borderline face joining the wrong pile drags a whole
# person's photographs in with it, because the centroid then sits between two
# people and attracts both.
#
# Lowered from 0.5 once the quality gates went in, and that order matters. With
# every face accepted, different people reached 0.477 and the threshold had to
# sit above that. With profiles, blurs and tiny faces rejected, the same set of
# strangers tops out at 0.229 and averages 0.104. Cleaning the input is what
# buys the room to be more generous about matching, which is what catches a face
# that has changed rather than only one photographed twice in a day.
#
# It still errs towards splitting, which is the recoverable direction: a split
# is a five second fix with the merge button, while a wrong merge quietly loses
# the distinction between two people and nothing on screen says so.
MATCH = float(os.environ.get("JSONCAM_FACE_MATCH", "0.42"))

# Below MATCH but above this, the two are worth a second opinion rather than a
# guess in either direction. That band is where a person who has aged lives: the
# embeddings no longer agree, but they are not strangers either. A vision model
# is asked to look at the two faces, and "unsure" leaves them apart.
ADJUDICATE = float(os.environ.get("JSONCAM_FACE_ADJUDICATE", "0.26"))

# How close a face has to be to an existing look to be folded into it, rather
# than becoming a new look of the same person. Comfortably above MATCH: joining
# a look should mean "this is the same face again", while the gap between MATCH
# and this is "recognisably them, but they have changed".
SAME_LOOK = float(os.environ.get("JSONCAM_FACE_SAME_LOOK", "0.62"))

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


def find(image, keep_rejected=True):
    """Every face in a PIL image, with the unusable ones marked rather than dropped.

    Returns dicts with a normalised bounding box, a unit-length 128-float
    embedding, a small JPEG crop and the detector's confidence. A face that
    fails a gate carries `rejected` naming which one, and must never be
    clustered: its embedding describes a silhouette or a blur, not a person.

    They are kept anyway, because "the model could not see this face" and "there
    is nobody there" are different things, and only the first one is worth
    offering to a person who can see perfectly well who it is. The gates decide
    what the machine acts on; they should not decide what somebody is allowed to
    correct.

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
            edge = min(box_w, box_h)

            # YuNet hands back five landmarks after the box: right eye, left
            # eye, nose tip, then the two mouth corners.
            right_eye, left_eye, nose = row[4:6], row[6:8], row[8:10]
            eye_gap = float(np.linalg.norm(left_eye - right_eye))
            eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2
            yaw = abs(float(nose[0]) - eye_mid_x) / max(eye_gap, 1e-6)

            top, left = max(0, int(y)), max(0, int(x))
            patch = bgr[top:int(y + box_h), left:int(x + box_w)]
            if patch.size == 0:
                continue
            sharpness = float(cv2.Laplacian(
                cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

            rejected = None
            if edge < MIN_EDGE:
                rejected = "too small"
            elif eye_gap / max(edge, 1e-6) < MIN_EYE_RATIO:
                rejected = "not facing the camera"
            elif yaw > MAX_YAW:
                rejected = "turned away"
            elif sharpness < MIN_SHARPNESS:
                rejected = "too blurred"
            if rejected and not keep_rejected:
                continue

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
                "edge": int(edge),
                # Which gate this failed, or None when it is usable. Anything
                # with a reason here must not be clustered.
                "rejected": rejected,
                "yaw": round(yaw, 3),
                "sharpness": round(sharpness, 1),
                # One number for "how much should this face be trusted", used to
                # pick the picture that represents a person and to decide which
                # faces are allowed to start a new cluster.
                "quality": round(min(1.0, edge / 160) * min(1.0, sharpness / 400)
                                 * (1.0 - min(yaw, MAX_YAW) / MAX_YAW * 0.5) * score, 4),
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
