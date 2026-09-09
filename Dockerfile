# json-camera on a Hugging Face Space (docker SDK).
#
# The Space runs the real web app rather than the Gradio demo, so what is
# published is the same landing page, compressor and decompressor that run
# locally.
#
# CPU-only torch on purpose: the CUDA wheels are about 2 GB and a free Space has
# no GPU to point them at, so the default index would spend the whole image
# budget on kernels that never run.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    TMPDIR=/tmp

# Spaces run the container as uid 1000, which cannot write to a root-owned tree.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-space.txt

COPY --chown=user jsoncam ./jsoncam
COPY --chown=user web ./web
# Served at /AGENTS.md, so it has to be in the image. It lives at the repo root
# by convention, which is outside everything else copied here.
COPY --chown=user AGENTS.md ./AGENTS.md
# Only the slim, shippable checkpoints. checkpoints/ itself holds training
# checkpoints that carry Adam state at three times the size and are useless for
# inference, so copying the whole directory would triple the image for nothing.
COPY --chown=user checkpoints/stable ./checkpoints

# The two face models, about 38 MB, fetched at build time and verified by
# sha256. Not in git, because weights do not belong there; not fetched at
# runtime, because a machine that starts up without network access should still
# be able to find faces. A hash mismatch fails the build on purpose: a changed
# model would silently stop matching every embedding already stored, and a
# library that had learned who six people were would re-cluster them into
# strangers.
COPY --chown=user scripts/get_face_models.py ./scripts/get_face_models.py
RUN python scripts/get_face_models.py /home/user/app/checkpoints/faces

USER user

# PORT is read at runtime, not baked in: a Space expects 7860 and Cloud Run
# injects its own, so the same image serves both.
ENV JSONCAM_MODELS=/home/user/app/checkpoints \
    JSONCAM_FACE_MODELS=/home/user/app/checkpoints/faces \
    HOST=0.0.0.0 \
    PORT=7860

EXPOSE 7860
CMD ["python", "web/server.py"]
