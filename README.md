# json-camera

A learned image codec that stores a photograph as a JSON file.

An image goes in, a convolutional network crushes it down to a small grid of
integers, a learned entropy coder packs those integers into the fewest bits
they can honestly be written in, and the result lands in a `.json` file. A
second network reads that grid back and rebuilds the picture.

Nothing about the transform is hand-designed. The network *learns* what to keep
and what to throw away, by being trained against a loss that counts the output
file's size in bits.

## The idea, in the order it actually happens

**1. Images are numbers.** A 4K photo is 3840 x 2160 x 3 = ~25 MB of raw bytes.
Neighbouring pixels are almost always similar, so most of those bytes are
redundant.

**2. Strided convolutions squeeze it.** This is the part you already had the
intuition for. A convolution slides a small window over the image; giving it
`stride=2` means it hops two pixels at a time, so the output is half the size in
each direction. Four of them in a row shrink 3840x2160 down to a 240x135 grid.

The difference from the pooling you learned about is that pooling uses a *fixed*
rule (take the max, take the mean). Here the window's weights are **learned**.
Training discovers a rule that beats "take the mean", because it is optimised
for exactly one thing: being reconstructable later.

**3. Quantisation.** The latent grid is rounded to whole numbers. This is the
lossy step, and it is where most of the saving comes from. It is also not
differentiable, so during training we add uniform random noise instead of
rounding, which is a smooth stand-in that lets gradients through.

**4. The entropy model counts the bits.** A second small network learns, per
channel, how the values in that channel are distributed. Once you know a
value's probability you know its cost: `-log2(p)` bits. Common values cost a
fraction of a bit, rare ones cost more.

This is the piece that makes the whole thing work, and it is the piece most
people skip. Without it you are just saving an array. With it, the network
gets a **differentiable estimate of the file size** and can be trained to
minimise it directly.

**5. rANS packs the bits.** The range coder takes those probabilities and
writes the symbol stream at essentially the theoretical limit (measured: within
1.6% of Shannon entropy on this implementation).

**6. JSON.** The bitstream is base85-armoured into a text payload with a small
readable header.

## The loss function is the whole trick

```
loss = lambda * distortion + rate
```

`rate` is real bits per pixel from step 4. `distortion` is how wrong the
reconstruction is. Both are differentiable, so backprop pushes the encoder
toward representations that are simultaneously **cheap to write down** and
**sufficient to rebuild the image**. Those two goals fight each other, and
`lambda` decides who wins.

`lambda` is the quality knob and it is the only difference between a 200 KB
model and a 2 MB one. **Train one model per quality level.**

## Install

```bash
cd ~/dev/json-camera
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Use

```bash
# 1. get training images -- 800 DIV2K photos, ~3.5 GB.
#    (The canonical ETH host is usually throttled to a few KB/s, so this pulls
#    from a mirror instead.  Or skip it and point --images at your own folder.)
.venv/bin/python scripts/get_data.py

# 2. chop them into training patches
.venv/bin/jsoncam prepare --images data/DIV2K_train_HR --out data/patches.npy

# 3. train (Apple Silicon GPU is used automatically).
#    --val-cache is held-out data; the .best.pt checkpoint is chosen on THAT
#    loss, so you do not ship whichever epoch overfit hardest.
.venv/bin/jsoncam train --cache data/patches.npy --val-cache data/val_patches.npy \
    --out checkpoints/jc.pt --lmbda 0.0067 --epochs 10

#    or just: ./scripts/run_training.sh

# 4. shoot a photo into JSON, and back out
.venv/bin/jsoncam encode photo.jpg -o photo.json
.venv/bin/jsoncam decode photo.json -o restored.png

# 5. score it honestly against JPEG at the same file size.
#    Reports PSNR and MS-SSIM; MS-SSIM tracks what the picture actually looks
#    like far better at these bitrates.
.venv/bin/jsoncam eval photo.jpg
```

## What the file looks like

```json
{
  "format": "json-camera/1",
  "model": {"hidden": 128, "latent": 192, "fingerprint": "9a2c7f4c3efff0d9"},
  "image": {"width": 3840, "height": 2160},
  "latent": {"channels": 192, "height": 135, "width": 240},
  "codec": {"kind": "rans", "precision": 12, "lanes": 512, "count": 6220800},
  "payload": {"encoding": "b85", "data": "…"}
}
```

## Three things worth being straight about

**The model weights are part of the file format.** The decoder cannot rebuild
anything without the exact weights that encoded it — that is why every file
carries a `fingerprint` and decoding refuses on a mismatch. Your ~10 MB
checkpoint is a shared codebook, like the Huffman tables baked into every JPEG
decoder. It is amortised across every image you ever encode, but if you send
someone a `.json` and not the checkpoint, they get nothing.

**This is compression, not encryption.** It looks unreadable, but there is no
key and no secret — anyone with the checkpoint reads it. Do not use it to hide
anything. (If you want secrecy, encrypt the payload afterwards; the two are
separate jobs and mixing them makes both worse.)

**JSON costs 25%.** Text can only hold ~6.1 bits per character with base85, so
the payload is 25% bigger than the raw bitstream (base64 would be 33%). That is
the price of the container being JSON, not a flaw in the codec. `eval` prints
both numbers so you can always see the tax.

## Where the target lands

A 4K frame is 8,294,400 pixels, so the file size you want maps straight onto a
bitrate:

| target `.json` | bits per pixel | bitstream |
|---|---|---|
| 500 KB | 0.49 bpp | 400 KB |
| 250 KB | 0.25 bpp | 200 KB |
| 200 KB | 0.20 bpp | 160 KB |

For reference JPEG needs roughly 1.5-2.0 bpp to look clean, so anything at or
below 0.5 bpp is aggressive — and it is exactly the range where learned codecs
beat JPEG by the widest margin, because a network trained at that budget learns
to *synthesise* plausible texture rather than smear it into blocks.

`lambda` is what picks the row. Whether the model actually lands there depends
on how long you train. `eval` will tell you the truth.

## Layout

```
jsoncam/model.py     encoder / decoder convnets, GDN, the loss
jsoncam/entropy.py   learned per-channel CDF; exports integer freq tables
jsoncam/rans.py      vectorised interleaved range coder
jsoncam/codec.py     JSON container, tiling for large images
jsoncam/train.py     training loop
jsoncam/data.py      patch cache + dataset
jsoncam/cli.py       prepare / train / encode / decode / eval
```
