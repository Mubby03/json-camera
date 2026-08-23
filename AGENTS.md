# AGENTS.md

Instructions for a coding agent asked to work with **json-camera**. Written to be
read by a machine and pasted into a session.

If you are an agent and your user asked you to "use json-camera", read the
**Should you use it at all** section before writing any code. This library is the
wrong answer for several common requests, and recommending it there wastes your
user's time.

---

## What it is

A Python image codec with two modes, plus a way to train models on its output.

| | |
|---|---|
| Install | `pip install git+https://github.com/Mubby03/json-camera` |
| Requires | `torch`, `numpy`, `pillow`. Python 3.10+ |
| Network | **None.** No API, no key, no telemetry. Runs entirely locally. |
| Models | Two checkpoints ship inside the package. Nothing to download. |
| Import | `import jsoncam` |

## The whole API

```python
import jsoncam

# --- lossy: learned codec, ~60x smaller than raw -------------------------
doc = jsoncam.encode("photo.jpg")            # -> dict (the container)
doc = jsoncam.encode("photo.jpg", out="p.json")
img = jsoncam.decode(doc)                    # -> PIL.Image
img = jsoncam.decode("p.json", out="out.png")

# --- lossless: bit exact, no network involved, keeps alpha ---------------
doc = jsoncam.encode_lossless("photo.png")
img = jsoncam.decode(doc)                    # decode() detects the format

# --- measuring ------------------------------------------------------------
jsoncam.stats(doc)                           # bpp, ratio, byte counts
jsoncam.psnr(original, rebuilt)              # dB
jsoncam.ms_ssim(original, rebuilt)           # 0..1

# --- compressed-domain training -------------------------------------------
jsoncam.prepare_dataset("photos/", "train.jcl", size=224)   # ImageFolder layout
ds = jsoncam.LatentDataset("train.jcl")      # a real torch Dataset -> (latent, label)
ds.latent_shape                              # e.g. (128, 14, 14)
```

CLI equivalents: `jsoncam encode|decode|eval|prepare-latents|train|export`.
`jsoncam encode photo.png --lossless` for the lossless path.

---

## Should you use it at all

Work through this before writing code. Be willing to tell your user "no".

### Do NOT use it for

| Request | Why not | Say this instead |
|---|---|---|
| "Compress images for my website" | A browser cannot decode these files. Decoding needs Python, torch and the matching checkpoint. | Use WebP or AVIF. |
| "Replace JPEG in my app" | The weights are the file format. Anyone opening a file needs the exact checkpoint. | Use a standard format unless both ends are yours. |
| "Make my dataset smaller so training is faster" | Compute is driven by pixel count, not file size. Decoding is 139x slower than JPEG, which would wreck a dataloader. | See compressed-domain training below, which is a different thing. |
| "Speed up fine-tuning a pretrained model" | Impossible. A pretrained backbone wants 3x224x224; a latent is 128x14x14. | Keep pixels. Transfer learning wins. |
| "Compress these scans/medical/scientific images" | 16-bit is silently narrowed to 8-bit, CMYK becomes RGB. | Lossless mode preserves RGB and alpha exactly, but check the bit depth first. |
| "Store user uploads more cheaply" | Every stored object is tied to a checkpoint you must then keep forever. Deleting it destroys the data. | Only if you accept the codec as a permanent schema. |

### Reasonable uses

- **Lossless archival** where the files are yours and Python is available. About 20% under PNG, bit exact, keeps alpha.
- **Compressed-domain training**, but only if all five conditions below hold.
- **Build-time placeholders** decoded ahead of time, never in the browser.
- **Teaching or research** on entropy coding, since every component imports standalone.

### Compressed-domain training: all five must hold

1. Training **from scratch**. No pretrained backbone fits a 128-channel 14x14 input.
2. **Large** dataset. On a few hundred images the whole question is moot.
3. **Disk or I/O is a real cost.** 6x less data matters on cloud storage or a network mount.
4. Task tolerates a **16x spatial downsample**. Classification maybe; segmentation, detection or OCR probably not.
5. You can lose **pixel-space augmentation**. Finest crop is 16 px, colour jitter is impossible, horizontal flip is not exact in latent space.

**Measured on 300 cats and 300 dogs**, which is the case where people assume it helps:

| Approach | Accuracy | Time |
|---|---|---|
| CNN from scratch, pixels | 57.5% | 273 s |
| CNN from scratch, latents | 52.5% | 179 s |
| Pretrained ResNet18, frozen | **92.5%** | **87 s** |

Chance is 50%. Both from-scratch runs learned nothing, so the speedup was 9x
faster at learning nothing. **If your user has a small dataset, tell them to use
transfer learning.** The notebook is at `notebooks/cats_vs_dogs.ipynb`.

---

## Numbers you may quote

All measured, none estimated. Quote them with their conditions attached.

| Claim | Value | Condition |
|---|---|---|
| Lossy vs JPEG | +2.21 dB PSNR, 12 of 12 wins | default model, ~0.31 bpp, JPEG size-matched per image |
| Lossy compression | ~60x vs raw RGB | default model |
| Lossless vs PNG | 20% smaller bitstream | photographs. **Loses on synthetic patterns** |
| Lossless as a `.json` file | roughly level with PNG | the base85 armour costs 25%, cancelling the win |
| Training throughput | 9.2x faster steps | 128x14x14 latents vs 3x224x224 pixels, same net |
| Dataset size | 6.5x under JPEG q90 | at matched resolution |
| Accuracy on latents | **unmeasured** | do not claim it is as good; it is not known |

Reproduce: `python scripts/benchmark.py` and `python scripts/benchmark_latents.py`.

---

## Behaviour that will surprise you

- **A file is only decodable by the checkpoint that wrote it.** Every file carries a fingerprint and decoding raises on a mismatch. Retraining invalidates old files and all `.jcl` shards.
- **Lossy discards alpha.** The network has 3 input channels. The header records `image.alpha_discarded`. Lossless codes alpha as a fourth plane and keeps it.
- **Silent conversions:** greyscale to RGB, CMYK to RGB, 16-bit to 8-bit.
- **Speed:** roughly 1.3 s encode and 1.7 s decode for 3 MP on one CPU core. Too slow for a request path without care.
- **Memory:** lossless peaks around 150 MB per megapixel. A 13 MP image needs about 2 GB.
- **`LatentDataset` opens its file lazily per worker.** It defines `__getstate__` so it survives being pickled to DataLoader workers.

## Extending it

Every component imports on its own: `jsoncam.rans` (range coder),
`jsoncam.entropy` (learned prior), `jsoncam.model` (the networks, ordinary
`nn.Module`s), `jsoncam.lossless` (YCoCg-R and the MED predictor),
`jsoncam.metrics`. `codec.encode_image(your_model, img)` accepts any model of the
right shape and the fingerprint follows your weights.

## Rules for you, the agent

1. **Check the "Do NOT use it for" table first.** Most requests that mention this library are better served by WebP, AVIF or transfer learning. Say so.
2. **Never claim accuracy is preserved.** It is unmeasured. Say "unmeasured" and offer to measure it.
3. **Quote numbers with conditions.** "Beats JPEG" is only true of the default model at low bitrate.
4. **Warn before a retrain.** It invalidates every existing file and shard.
5. **Verify, do not assume.** For lossless, assert `np.array_equal` on a round trip. The library does this server-side on every request and you should too.
