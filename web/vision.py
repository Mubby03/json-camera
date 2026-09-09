"""What is in the picture: a caption, some tags, and any text it contains.

This is what makes a library searchable. A photo archive you cannot search is a
shoebox, and the date and place only get you so far: "the photo of my passport"
and "that beach" are how people actually look for things.

Three decisions worth knowing about.

**It sends the embedded preview, not the photograph.**  Every container already
carries a 320 pixel WebP of itself, which is roughly a hundred tokens and plenty
to caption from.  Sending the full frame would cost fifteen times as much and
tell the model nothing it cannot already see.  The preview being there for the
gallery is what makes this nearly free.

**It runs once, in the background, and stores the answer.**  Never on a page
load, never on a refresh.  The row is written pending at upload and a worker
fills it in, so a restart resumes rather than redoing, and the gallery is always
just a query.

**It fails quietly.**  A photograph must still upload when the API is down, the
key is missing or the bill is unpaid.  Every failure here returns None and the
photo keeps its date, its place and its thumbnail.
"""

import base64
import json
import logging
import os

log = logging.getLogger("jsoncam.vision")

# Whether this model accepts output_config.effort. Not every model does: it is
# rejected outright on Haiku 4.5, which is exactly the model somebody switches
# to when they want this cheap. Rather than carry a compatibility matrix that
# goes stale, the first rejection is detected and remembered for the process.
_effort_ok = {}

# Opus 5 is the default because it is the best model, not because it is the
# cheapest. On the preview-sized images this sends, expect roughly $0.006 a
# photograph. Two ways down, both a deliberate choice rather than something to
# do silently: JSONCAM_VISION_MODEL=claude-haiku-4-5 is about five times cheaper
# (~$0.0012) and entirely adequate for captioning, and Sonnet 5 sits between
# them. Set the env var; nothing else changes.
MODEL = os.environ.get("JSONCAM_VISION_MODEL", "claude-opus-5")

# What the model is asked to fill in. `strict`-style schemas mean the answer
# parses without defensive string handling.
SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": "One plain sentence describing the photograph, as a person "
                           "would say it aloud. No preamble.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Between three and ten short lowercase keywords: subjects, "
                           "objects, setting, occasion. Single words or two at most.",
        },
        "text": {
            "type": "string",
            "description": "Any text legible in the image, verbatim. Empty string if none. "
                           "This is what makes a photographed document or receipt findable.",
        },
        "people": {
            "type": "integer",
            "description": "How many people are visible. 0 if none.",
        },
        "kind": {
            "type": "string",
            "enum": ["photo", "screenshot", "document", "receipt", "artwork", "other"],
            "description": "What sort of image this is.",
        },
    },
    "required": ["caption", "tags", "text", "people", "kind"],
    "additionalProperties": False,
}

PROMPT = (
    "Describe this photograph for someone searching their own photo library later. "
    "Be concrete and specific: name what is actually visible rather than describing "
    "the mood. If there is legible text, transcribe it, because that is often the "
    "only way the owner will find the picture again. Do not guess at names of people "
    "or at where it was taken."
)


def available():
    """Whether captioning can run at all, without spending a request to find out.

    The SDK resolves credentials from more than one place, and constructing a
    client proves nothing because it does not validate until the first call. So
    this checks the same sources the SDK does, in the same order: the two
    environment variables, then a profile written by `ant auth login`. An unset
    ANTHROPIC_API_KEY on its own does not mean there are no credentials.
    """
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    from pathlib import Path

    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "anthropic"
    return config.is_dir() and any(config.iterdir())


def describe(preview_bytes, media_type="image/webp"):
    """Caption one image. Returns a dict, or None if anything at all went wrong.

    `preview_bytes` is the small embedded thumbnail, not the full photograph.
    """
    if not preview_bytes or not available():
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(preview_bytes).decode("ascii"),
                    },
                },
                {"type": "text", "text": PROMPT},
            ],
        }]
        schema = {"format": {"type": "json_schema", "schema": SCHEMA}}

        def ask(with_effort):
            config = dict(schema)
            if with_effort:
                # Captioning is a description task, not a reasoning one, so the
                # cheapest effort is also the right one here.
                config["effort"] = "low"
            return client.messages.create(model=MODEL, max_tokens=1024,
                                          output_config=config, messages=message)

        try:
            response = ask(_effort_ok.get(MODEL, True))
        except anthropic.BadRequestError as error:
            # Haiku 4.5 rejects `effort` outright. Learn that once and carry on
            # without it, rather than failing every caption on the cheap model.
            if "effort" not in str(error).lower() or not _effort_ok.get(MODEL, True):
                raise
            log.warning("%s rejected output_config.effort; retrying without it", MODEL)
            _effort_ok[MODEL] = False
            response = ask(False)

        # A safety decline is a real outcome on user-supplied photographs, and it
        # is not an error: the photo simply does not get a caption.
        if getattr(response, "stop_reason", None) == "refusal":
            log.info("caption declined by safety classifier")
            return None
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
    except Exception as error:
        # Quiet in the sense that the upload survives, not in the sense that
        # nobody can find out why every caption is missing. The first version of
        # this swallowed the reason and cost an afternoon.
        log.warning("captioning failed: %s: %s", type(error).__name__, error)
        return None

    tags = [str(t).strip().lower() for t in (data.get("tags") or []) if str(t).strip()]
    return {
        "caption": (data.get("caption") or "").strip() or None,
        "tags": tags[:12],
        "text": (data.get("text") or "").strip() or None,
        "people": int(data.get("people") or 0),
        "kind": data.get("kind") or "photo",
        "model": MODEL,
    }


def searchable(analysis, item=None):
    """One lowercase blob of everything worth matching a query against.

    Built here rather than at query time so searching is a single indexed lookup
    instead of a join across four columns and a JSON parse per row.
    """
    parts = []
    if analysis:
        parts += [analysis.get("caption") or "", " ".join(analysis.get("tags") or []),
                  analysis.get("text") or "", analysis.get("kind") or ""]
    if item:
        # The things a person also searches by, and which cost nothing to include.
        parts += [item.get("name") or "", item.get("camera") or "",
                  item.get("place") or "", (item.get("captured_at") or "")[:7]]
    return " ".join(p for p in parts if p).lower()
