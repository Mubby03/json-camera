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

# Two providers, because captioning a photo library is exactly the workload
# where the per-photo price is the whole argument. Measured on the preview-sized
# images this sends (roughly 250 tokens of prompt, 150 of answer):
#
#     claude-opus-5                 ~$0.0060   the best captions
#     claude-haiku-4-5              ~$0.0012
#     deepseek-v4-flash-vision-exp  ~$0.0005   peak, half that off-peak
#
# JSONCAM_VISION_PROVIDER picks one: "claude", "deepseek", or "auto". Auto
# prefers DeepSeek when its key is present, on the grounds that somebody who
# went and got a DeepSeek key wants it used.
PROVIDER = os.environ.get("JSONCAM_VISION_PROVIDER", "auto").strip().lower()

CLAUDE_MODEL = os.environ.get("JSONCAM_VISION_MODEL", "claude-opus-5")
DEEPSEEK_MODEL = os.environ.get("JSONCAM_DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp")
DEEPSEEK_URL = os.environ.get("JSONCAM_DEEPSEEK_URL",
                              "https://api.deepseek.com/chat/completions")

# Kept for anything that still reads it, and for the settings endpoint.
MODEL = CLAUDE_MODEL

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

SHAPE_INSTRUCTION = (
    "You describe photographs for a personal photo library. Reply with a single "
    "JSON object and nothing else, with exactly these keys: "
    '"caption" (one plain sentence), '
    '"tags" (array of 3 to 10 short lowercase keywords), '
    '"text" (any text legible in the image, verbatim, or an empty string), '
    '"people" (integer count of visible people, 0 if none), '
    '"kind" (one of "photo", "screenshot", "document", "receipt", "artwork", "other").'
)

PROMPT = (
    "Describe this photograph for someone searching their own photo library later. "
    "Be concrete and specific: name what is actually visible rather than describing "
    "the mood. If there is legible text, transcribe it, because that is often the "
    "only way the owner will find the picture again. Do not guess at names of people "
    "or at where it was taken."
)


def claude_ready():
    """Whether Claude could be called, without spending a request to find out.

    The SDK resolves credentials from more than one place and does not validate
    until the first call, so constructing a client proves nothing. This checks
    the same sources in the same order: the two environment variables, then a
    profile written by `ant auth login`. An unset ANTHROPIC_API_KEY on its own
    does not mean there are no credentials.
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


def deepseek_ready():
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def active():
    """Which provider will actually be used, or None.

    An explicit choice is honoured even when its key is missing, so that a
    misconfiguration shows up as "captions are off" plus a log line naming the
    provider, rather than quietly falling through to the one that costs more.
    """
    if PROVIDER == "claude":
        return "claude" if claude_ready() else None
    if PROVIDER == "deepseek":
        return "deepseek" if deepseek_ready() else None
    if deepseek_ready():
        return "deepseek"
    return "claude" if claude_ready() else None


def model_name(provider=None):
    return DEEPSEEK_MODEL if (provider or active()) == "deepseek" else CLAUDE_MODEL


def available():
    """Whether captioning can run at all."""
    return active() is not None


def describe(preview_bytes, media_type="image/webp"):
    """Caption one image. Returns a dict, or None if anything at all went wrong.

    `preview_bytes` is the small embedded thumbnail, not the full photograph.
    Both providers are coerced into the same shape, because the search index and
    the gallery must not be able to tell which one answered.
    """
    if not preview_bytes:
        return None
    provider = active()
    if provider is None:
        return None
    try:
        if provider == "deepseek":
            data = _ask_deepseek(preview_bytes, media_type)
        else:
            data = _ask_claude(preview_bytes, media_type)
    except Exception as error:
        # Quiet in the sense that the upload survives, not in the sense that
        # nobody can find out why every caption is missing. The first version of
        # this swallowed the reason and cost an afternoon.
        log.warning("captioning failed via %s: %s: %s",
                    provider, type(error).__name__, error)
        return None
    if data is None:
        return None
    return _coerce(data, model_name(provider))


def _coerce(data, model):
    """One shape, whatever answered.

    Both providers are asked for the same fields, but only Claude's schema is
    enforced by the API. DeepSeek is asked in the prompt and could return a
    string where a number belongs, so everything is normalised here rather than
    trusted, and the caller cannot tell the difference.
    """
    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [part.strip() for part in tags.split(",")]
    tags = [str(t).strip().lower() for t in tags if str(t).strip()]

    people = data.get("people")
    try:
        people = int(people)
    except (TypeError, ValueError):
        people = 0

    kind = str(data.get("kind") or "photo").strip().lower()
    if kind not in ("photo", "screenshot", "document", "receipt", "artwork", "other"):
        kind = "other"

    return {
        "caption": (str(data.get("caption") or "").strip() or None),
        "tags": tags[:12],
        "text": (str(data.get("text") or "").strip() or None),
        "people": people,
        "kind": kind,
        "model": model,
    }


# Whether this Claude model accepts output_config.effort. Not every model does:
# it is rejected outright on Haiku 4.5, which is exactly the model somebody
# switches to when they want this cheap. Rather than carry a compatibility
# matrix that goes stale, the first rejection is remembered for the process.
_effort_ok = {}


def _ask_claude(preview_bytes, media_type):
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
        return client.messages.create(model=CLAUDE_MODEL, max_tokens=1024,
                                      output_config=config, messages=message)

    try:
        response = ask(_effort_ok.get(CLAUDE_MODEL, True))
    except anthropic.BadRequestError as error:
        if "effort" not in str(error).lower() or not _effort_ok.get(CLAUDE_MODEL, True):
            raise
        log.warning("%s rejected output_config.effort; retrying without it", CLAUDE_MODEL)
        _effort_ok[CLAUDE_MODEL] = False
        response = ask(False)

    # A safety decline is a real outcome on user-supplied photographs, and it is
    # not an error: the photo simply does not get a caption.
    if getattr(response, "stop_reason", None) == "refusal":
        log.info("caption declined by safety classifier")
        return None
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _ask_deepseek(preview_bytes, media_type):
    """DeepSeek's vision model, over its OpenAI-shaped chat endpoint.

    Raw HTTP on purpose: this is one POST with a JSON body, and urllib is
    already here for the place lookups. Adding an SDK for it would be a
    dependency to carry for no gain.

    Unlike Claude there is no server-side schema enforcement here, so the shape
    is requested in the prompt, `json_object` is asked for, and the answer is
    normalised by `_coerce` rather than trusted.
    """
    import urllib.error
    import urllib.request

    data_url = f"data:{media_type};base64,{base64.b64encode(preview_bytes).decode('ascii')}"
    body = json.dumps({
        "model": DEEPSEEK_MODEL,
        "max_tokens": 1024,
        # Low creativity: this is description, not writing.
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SHAPE_INSTRUCTION},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                {"type": "text", "text": PROMPT},
            ]},
        ],
    }).encode("utf-8")

    request = urllib.request.Request(
        DEEPSEEK_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
                 "Content-Type": "application/json",
                 "User-Agent": "json-camera/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # The body carries the actual reason (bad key, no credit, bad model id),
        # and without it every failure looks the same in the log.
        detail = error.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from None

    text = payload["choices"][0]["message"]["content"]
    return json.loads(_unfence(text))


def _unfence(text):
    """Strip a ```json fence if one came back.

    `json_object` mode should prevent this, but the model is experimental and a
    fenced block is the single most common way a chat model returns JSON.
    """
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


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
