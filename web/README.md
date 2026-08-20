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

## Deploying

Two targets, one Dockerfile. `PORT` is read at runtime rather than baked in, so
the same image serves a Space on 7860 and Cloud Run on whatever it injects.

### Google Cloud Run

```bash
./scripts/deploy_cloudrun.sh
```

Cloud Run builds the image itself, so no local docker daemon is needed.
`.gcloudignore` keeps the build context at about 8 MB; without it Cloud Build
would upload the 11 GB of training data sitting in `data/`.

The defaults in the script are deliberate. 512Mi OOMs on a 3 megapixel image and
the failure looks like a bare 503, so it asks for 2Gi. Concurrency is 2 because
encoding is CPU bound for seconds at a time and stacking requests on one
instance only makes everyone wait. `--max-instances` is a hard cap so a burst
cannot run up a bill. `--min-instances 0` is what keeps it inside the free tier,
at the cost of a cold start of roughly half a minute while torch loads.

### Hugging Face Space

```bash
.venv/bin/python scripts/deploy_space.py --repo you/json-camera
```

Note that Hugging Face now requires a PRO subscription for Docker and Gradio
Spaces. Only static Spaces are free, so this path costs money as of August 2026.
