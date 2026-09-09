"""Teach Pillow to open the format iPhones actually shoot.

Every photograph taken on a modern iPhone is HEIC, and Pillow does not read HEIC
on its own.  Without this, the entire phone story fails at the first step: the
Shortcut uploads a real photograph and the server answers "that does not look
like an image we can read", which is true and useless.

`pillow-heif` is an optional dependency rather than a hard one, because the
library itself is a codec and should not require a second codec to import.  So
registration is a call somebody makes, not a side effect of importing anything,
and it says plainly whether it worked.
"""

_state = None


def enable_heif():
    """Register the HEIF/HEIC opener with Pillow. Safe to call more than once.

    Returns True if Pillow can now open HEIC, False if the optional dependency
    is missing. Callers that care should say so in their own output rather than
    let a phone photo fail later with a confusing message.
    """
    global _state
    if _state is not None:
        return _state
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _state = True
    except Exception:
        # An old or partial install should degrade to "JPEG and PNG still work",
        # not take the process down on import.
        _state = False
    return _state


def heif_available():
    """Whether HEIC can be opened, without registering anything."""
    return bool(_state)
