# json-camera — full project brief

*Reference document. Every number below was measured during development, not
estimated. Where something is an estimate or still in progress it says so.*

- **Live demo:** https://json-camera.fly.dev
- **Source:** https://github.com/Mubby03/json-camera
- **Write-up:** https://www.mubby.space/projects/json-camera
- **Built by:** Mubaraq Lawal (mubby.space)

---

## 1. What it is, in one paragraph

An image codec built from scratch that stores a photograph as a JSON file. A
convolutional network compresses the picture to a small grid of whole numbers, a
second network learns what those numbers cost in bits, a range coder packs them
at near the theoretical limit, and the result is written as readable JSON. A
third network reads that file and rebuilds the photograph. Nothing about the
transform is hand-designed: it was trained against a loss that counts the output
file's size in bits, so it learned what to keep and what to discard by being
scored on the answer.

It has since grown a second, lossless codec that uses no network at all, and a
Python library that lets other people train models directly on its compressed
output rather than on pixels.

---

## 2. The three things it does

### 2.1 Lossy codec (the learned one)

Six stages: strided convolutions downsample 16x → quantisation → a learned
entropy model prices every value → rANS range coder packs them → base85 text →
JSON container.

The whole idea is one loss function:

```
loss = lambda * distortion + rate
```

`rate` is real bits, estimated by the entropy model. Because that term is
differentiable, backpropagation optimises **file size directly** rather than a
proxy for it. Most approaches train a network to make images look good and hope
a general-purpose compressor does well on the output; this one is scored on the
bytes. `lambda` is the quality knob and the only difference between a small file
and a large one.

**Two models ship, at different points on the rate-distortion curve. Both were
benchmarked on 12 held-out DIV2K photographs with JPEG size-matched per image via
binary search.**

*Small model (lambda 0.0067), mean 0.31 bpp:*

| | json-camera | JPEG | delta |
|---|---|---|---|
| PSNR | 29.63 dB | 27.42 dB | **+2.21 dB** |
| MS-SSIM | 14.33 dB | 10.86 dB | **+3.46 dB** |

**Won 12 of 12 images on both metrics.** Range +0.10 to +3.69 dB — weakest on
dense detail, strongest on smooth scenes.

*Sharp model (lambda 0.05), mean 1.03 bpp:* 32.84 dB against JPEG's 33.92, so
**−1.08 dB and only 2 wins of 12.** Absolute quality is 2.7 dB better than the
small model (29.43 dB held-out against 26.76), but it loses the comparison.

**This is the honest and more interesting result.** Learned codecs beat JPEG
hardest at low bitrate, where JPEG collapses into blocks. Around 1 bpp JPEG is in
its comfort zone, and a 2M-parameter model trained for three hours on 800 images
does not keep up. Turning up quality made a better picture and a worse codec.
Both ship, and the picker names the cost of each.

Matched size is the load-bearing part of the methodology: JPEG quality is
binary-searched to land on the same byte count, so neither codec gets a
flattering operating point. That is also why the second result could not be
hidden.

### 2.2 Lossless codec (no network at all)

Added when the requirement became "discard nothing". A neural encoder earns its
ratio by deciding what to drop; when nothing may be dropped the job is pure
prediction. Three reversible steps:

1. **YCoCg-R** — a lifting colour transform that decorrelates RGB using only
   integer adds and shifts, so it inverts exactly. Worth ~1.7 MB on a 3 MP photo
   before anything else happens.
2. **MED** — the median edge predictor from JPEG-LS. Guesses each pixel from its
   left, up and up-left neighbours, switching behaviour at edges.
3. **rANS** over the prediction errors, with frequency tables measured per image.

**The interesting engineering problem:** decoding looks strictly sequential,
because each pixel needs its neighbours already rebuilt, and three million
sequential Python iterations is not a codec. But MED only ever looks *left and
up*, so every pixel on an anti-diagonal depends solely on earlier diagonals — a
whole diagonal resolves in one vectorised step. That turns **3.1 million scalar
operations into 3,575 array ones**, and a full 3 MP reconstruction takes about
half a second.

**Measured, verified bit-exact by round-trip on every request:**

| format | size | exact? |
|---|---|---|
| PNG | 4.89 MB | yes |
| **json-camera lossless** | **3.98 MB (20% smaller)** | yes |
| WebP lossless | 3.76 MB | yes |

Two honest caveats, both stated on the site: the JSON armour costs 25%, which
cancels the win almost exactly (as a `.json` file it lands level with PNG). And
it loses on synthetic content — PNG runs filtered rows through zlib and finds
*exact repeats*, beating this coder by ~50% on a sine pattern. This is a
photograph coder with no match model.

### 2.3 Compressed-domain training library (the newest, and the most useful)

The pitch most people would make — "compress your dataset to save compute" — is
false, and measuring it proved it. CNN compute is driven by **pixel count**, not
file size, and json-camera decodes **139x slower than JPEG** (45 ms vs 0.33 ms),
which would wreck any dataloader.

But splitting that 45 ms apart showed where it goes:

| | |
|---|---|
| unpack the integers (rANS) | 4.9 ms |
| run the decoder network | 19.7 ms ← all the cost |

**So don't decode.** Train the model on the latent grid directly.

| | tensor | throughput | disk per image |
|---|---|---|---|
| pixels | 3 x 224 x 224 = 150,528 | 50 img/s | 20.1 KB (JPEG q90) |
| **latents** | **128 x 14 x 14 = 25,088** | **462 img/s** | **3.1 KB** |

**9.2x faster training steps and 6.5x less disk**, same architecture, same batch.
For ImageNet-scale that is 22.4 GB → 3.6 GB. Reproducible by anyone:
`python scripts/benchmark_latents.py`.

```python
import jsoncam
jsoncam.prepare_dataset("photos/", "train.jcl")   # ImageFolder layout works as-is
ds = jsoncam.LatentDataset("train.jcl")           # drop into DataLoader
```

**Design decision made from measurement, not taste:** raw int16 latents load
instantly but are **2.7x larger on disk than JPEG**, which destroys half the
pitch. So shards store the rANS bitstream and workers unpack it. At ~5 ms each,
four workers deliver ~750 img/s while the model consumes 462 — decoding is free
because it happens off the critical path.

---

## 3. Architecture and scale

- **1,944,771 parameters** — encoder 969,632, decoder 969,507, entropy model 5,632
- Shipped checkpoint: **7.8 MB**
- Trained on **800 DIV2K photographs**, validated on **64 held-out** ones
- 38,400 training patches of 256x256, 1,024 held-out patches
- ~4,000 lines of Python, 1,279 lines of web front end, 1,303 lines of monitor
- **29 tests**

Components: analysis/synthesis convnets with GDN activations, a factorised prior
entropy model with monotonic per-channel CDFs, a vectorised interleaved rANS
coder (512 lanes), tiled encode/decode with margins so seams do not show, and a
JSON container carrying a model fingerprint.

---

## 4. Engineering findings worth telling

These are the parts that make a good post, because each one is a specific bug or
wrong assumption caught by measuring rather than assuming.

**The quality metric cost 9x more than the codec it was measuring.** Profiling a
request: encode 0.76s, decode 0.61s, previews 0.13s, JPEG search 0.02s, and
**MS-SSIM 6.21s** — running twice per request. The filter is a Gaussian, which is
separable, so two 1D passes replaced one 11x11 convolution: 22 multiply-adds per
pixel instead of 121. **6.21s → 1.24s**, outputs agreeing to 5e-07. End-to-end
request time went from 17.1s to 5.6s.

**A shared vCPU is not a small CPU, it is a fraction of one.** First deploy used
`shared-cpu-1x`, matching the other apps on the account. It 502'd. Measured
inside the machine: **56 seconds just to import torch**, 6.2s to encode a 0.2 MP
image — roughly **85x slower than a laptop core**. `performance-1x` brought that
same encode to 0.24s.

**Lossless killed the server on a 13 MP upload.** Not an error — a restart. The
kernel OOM-killed the process and every in-flight request died with it. Measured
peak **2963 MB against a 2048 MB machine**. Instrumenting each stage showed the
arrays were twice the width they needed: int64 for values that are small table
indices, and the range coder was building three whole-stream arrays that existed
only to be read one row at a time (933 MB). Now int32 throughout with
intermediates freed: **1914 MB, 35% less**, plus a hard cap so it refuses
politely instead of taking everyone down.

**Two bugs were destroying photos before the codec even ran.** EXIF orientation
was never applied, so phone photos encoded sideways. And the ICC colour profile
was dropped on output — phones shoot Display P3, we kept the P3 numbers and threw
away the tag saying so, so viewers painted them as sRGB. On a muted test image
that mismatch alone was a **40 dB error**; on a saturated photo far worse. Both
now ride through the container.

**"Lossless" flags lie.** AVIF reported 2.07 MB — 58% better than PNG, which
would have been a headline. Round-trip verification showed **max pixel difference
47, mean 1.12**. It is a lossy file wearing a lossless label. Verified
round-trip is the only measurement that counts.

**A DataLoader pickles your dataset.** Once `__getitem__` opened a file handle,
the dataset became unpicklable and every multi-worker run crashed. Fixed with
`__getstate__`, and there is a test for that exact sequence — read first, then
pickle — because that is the order that breaks.

---

## 5. Ideas tested and disproven

Three proposals were investigated properly and measured rather than dismissed.

**Recursive compression** ("keep compressing until it is tiny"). Ran it: gzip
round 1 took 4,000,000 bytes to 3,249,006. **Every round after that grew it** by
~1,000 bytes. The reason is counting, not engineering: there are 16,777,216
possible 24-bit inputs but only 8,388,608 possible 23-bit outputs, so any scheme
that shrinks some inputs must grow others.

**Prime factorisation** ("32 = 2^5 is shorter"). Encoded 4,000 random 24-bit
numbers both ways: 96,000 bits as plain numbers, **173,880 bits as
factorisations — 1.81x bigger.** The file must carry which primes, which
exponents, and how many. Worst case cost 39 extra bits, and a large prime cannot
be factored at all.

**2D Gaussian splatting.** Implemented it, fitted it to a real photograph, scored
it honestly as `splat + residual`:

| method | splat bits | residual bits | total |
|---|---|---|---|
| MED (shipped) | 0.00 | 2.99 | **2.99** |
| 200 Gaussians (30.0 dB) | 0.59 | 4.87 | 5.46 |
| 3,200 Gaussians (41.5 dB) | 9.38 | 2.93 | **12.30** |

Worse at every setting, and worse the more you add — at 3,200 Gaussians the file
exceeds raw pixels. Splatting works in 3D because hundreds of views amortise each
Gaussian; a single photo has nothing to amortise against. MED wins because its
model costs **exactly zero bytes**.

**The underlying reason all three fail:** a photograph's grain — sensor noise,
demosaic, fine texture — was measured at **4.60 bits per sample**, needing 5.41
MB for this image alone. Random data does not compress. To go below the lossless
floor you must discard the grain, which is what "lossy" means.

---

## 6. What is deployed

| | |
|---|---|
| **Web app** | FastAPI + custom front end, no framework, on Fly (performance-1x, 2 GB, jnb, scales to zero) |
| **Pages** | landing, compressor, decompressor, 404 |
| **Design** | flat, sharp-edged, zero rounded corners, Helvetica, dark, mint accent |
| **Features** | drag and drop, live JPEG comparison, byte-by-byte accounting, JSON header inspection, original filenames preserved through the round trip |
| **Monitor** | separate live training dashboard, standard library only, SVG charts, 16 parsing self-checks |
| **Portfolio** | project page on mubby.space with embedded lazy-loading demo |

**Filenames survive both directions:** `Lagos Rooftops.png` → `Lagos
Rooftops.json` → `Lagos Rooftops.png`. The original name is stored inside the
container, so renaming the `.json` on disk does not lose it.

---

## 7. In progress at time of writing

A higher-quality model is training (lambda 0.05 vs the original 0.0067), aimed at
the "small file that looks identical" operating point. Epoch 2 of 10:

| epoch | val PSNR | val bpp |
|---|---|---|
| 1 | 23.05 dB | 0.866 |
| 2 | 25.04 dB | 0.947 |

The v1 model finished at 26.76 dB at 0.346 bpp. This one is spending ~2.7x the
bits. Target is 38-40 dB, which is where a side-by-side becomes
indistinguishable. **That target is an estimate, not yet a measurement.**

---

## 8. Honest limitations

Worth including in any post, because stating them is more credible than not.

- **The weights are the file format.** A `.json` is only decodable by the exact
  checkpoint that encoded it. Every file carries a fingerprint and decoding
  refuses on mismatch. Send someone a file without the model and they get nothing.
- **This is compression, not encryption.** No key, no secret. Anyone with the
  checkpoint reads it.
- **JSON costs 25%.** Base85 carries ~6.1 bits per character. That is the price of
  the container being text, not a flaw in the codec. Both numbers are always shown.
- **One quality level shipped so far.** Lambda picks one point on the
  rate-distortion curve; the shipped model sits at the aggressive end.
- **Encode/decode is CPU-bound**, ~1.3s/1.7s for 3 MP on one dedicated core. The
  demo caps uploads because of this, not because the codec cannot handle more.
- **Lossless is capped at 10 MP on the server** (~150 MB working memory per
  megapixel). The CLI has no cap.

---

## 9. Suggested angles for a post

1. **"I made CNN training 9x faster by never decoding the images."** Strongest
   hook: a measured number, a mechanism, and a script anyone can run to check.
2. **"My quality metric was 9x more expensive than the codec it was measuring."**
   Very relatable profiling story with a clean mathematical fix.
3. **"Three compression ideas I tested and disproved."** Recursion, factorisation,
   Gaussian splatting — with the counting argument as the punchline.
4. **"'Lossless' flags lie: always verify the round-trip."** Short, sharp,
   genuinely useful to anyone working with image pipelines.
5. **"I beat JPEG on 12 of 12 photographs — here is what that actually took."**
   The headline result, with matched-size methodology as the credibility anchor.

**Tone note:** the credible version of all of these includes the caveats. The
lossless mode ties with PNG once wrapped in JSON. The library only helps if you
never decode. The codec needs its own weights to open a file. Saying so is what
separates it from marketing.
