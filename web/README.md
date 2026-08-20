# Web app

The public face of json-camera: a landing page, a compressor and a decompressor.

```bash
.venv/bin/python web/server.py
```

Then open <http://localhost:8000>.

| Route | What it is |
|---|---|
| `/` | landing page, ends on the call to action |
| `/compress` | photograph in, `.json` out, with the full byte accounting |
| `/decompress` | `.json` in, photograph out, under its original name |
| `/api/models` | the checkpoints this server can offer |
| `/api/compress` | multipart upload, returns a summary and a result id |
| `/api/decompress` | multipart upload, returns a summary and a result id |
| `/api/preview/{id}/{original,decoded,jpeg}` | PNG previews |
| `/api/download/{id}` | the `.json` or the `.png`, with a real filename |

## Filenames

The original filename is written into the container at encode time, so the
round trip preserves it in both directions:

```
Lagos Rooftops.png  ->  Lagos Rooftops.json  ->  Lagos Rooftops.png
```

The decompressor reads the name out of the file rather than off the upload, so
renaming the `.json` on disk does not lose the original name. It is stored in
`image.name`, which older files simply do not carry, and those fall back to the
name of the upload.

## Why results are held server side

A 2 megapixel photograph makes a `.json` of a few hundred kilobytes, plus two
preview images. Returning all of that inline would stall the page on the JSON
parse, so a result is written to a temp directory under a random id and the
browser is handed a small summary. The heavy parts are then ordinary image and
file requests, which also means the download carries a real
`Content-Disposition` filename rather than a blob URL.

Results are swept after an hour. The sweep runs on write, so there is no timer
and nothing to supervise.

## Notes

- Uploads are capped at 40 MB and the long side is scaled to 3840 pixels. Both
  are environment variables: `JSONCAM_MAX_UPLOAD`, `JSONCAM_MAX_SIDE`.
- `JSONCAM_MODELS` picks the checkpoint directory, default `checkpoints/stable`.
- The quality picker reports the held-out bpp when a checkpoint carries one. The
  training figure is measured against the noise proxy and reads high, so quoting
  it would misdescribe every model in the list.
- On decompress the model is chosen by matching the fingerprint in the file. A
  mismatch is refused with an explanation rather than handed back as noise.
- The result id comes off the URL, so it is checked against the store root
  before any file is opened.
