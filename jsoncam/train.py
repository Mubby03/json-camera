"""Train the codec.

The whole idea lives in the loss:  lambda * distortion + rate.

`rate` is real bits, estimated by the entropy model.  `distortion` is how far
the reconstruction is from the original.  Because both terms are
differentiable, backprop pushes the encoder toward representations that are
simultaneously easy to describe cheaply AND sufficient to rebuild the image.
Nobody hand-designs the transform -- it falls out of that trade-off.

lambda is the quality knob and it is the ONLY thing separating a 200 KB file
from a 2 MB one.  Train one model per quality level.
"""

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .data import PatchDataset
from .model import JSONCamera, rate_distortion_loss


def psnr(mse):
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def pick_device(want):
    if want != "auto":
        return want
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model, dl, lmbda, device):
    """Honest pass over held-out data: real rounding, no augmentation."""
    model.eval()
    agg = {"loss": 0.0, "bpp": 0.0, "mse": 0.0, "n": 0}
    for x in dl:
        x = x.to(device, non_blocking=True)
        r = rate_distortion_loss(model(x), x, lmbda)
        agg["loss"] += r["loss"].item(); agg["bpp"] += r["bpp"].item()
        agg["mse"] += r["mse"].item(); agg["n"] += 1
    n = max(1, agg["n"])
    return agg["loss"]/n, agg["bpp"]/n, psnr(agg["mse"]/n)


def main(argv=None):
    ap = argparse.ArgumentParser("jsoncam-train")
    ap.add_argument("--cache", default="data/patches.npy")
    ap.add_argument("--val-cache", default=None,
                    help="held-out patches; best checkpoint is chosen on THIS loss")
    ap.add_argument("--out", default="checkpoints/jc.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lmbda", type=float, default=0.01,
                    help="quality knob: higher = better image, bigger file")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent", type=int, default=192)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    print(f"device: {device}")
    # Emit the run's own configuration. Anything reading this log later, the
    # monitor included, should learn what was run from the run itself rather
    # than from a script that may have moved on.
    print(f"config: lmbda={args.lmbda} hidden={args.hidden} latent={args.latent} "
          f"batch={args.batch} lr={args.lr} epochs={args.epochs} out={args.out}")

    ds = PatchDataset(args.cache)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    print(f"{len(ds)} patches, {len(dl)} steps/epoch")

    # Held-out set. Selecting "best" on training loss just picks the most
    # overfit epoch, which is exactly the checkpoint you do not want to ship.
    val_dl = None
    if args.val_cache and os.path.exists(args.val_cache):
        vds = PatchDataset(args.val_cache, augment=False)
        val_dl = DataLoader(vds, batch_size=args.batch, shuffle=False,
                            num_workers=0, drop_last=False)
        print(f"{len(vds)} held-out patches for validation")
    else:
        print("no --val-cache: selecting best on TRAIN loss (overfit risk)")

    model = JSONCamera(args.hidden, args.latent).to(device)
    start_epoch = 0
    # The entropy model's own parameters want a separate, larger step size --
    # they shape a distribution, not a feature map, and are slow to move otherwise.
    prior_params = list(model.prior.parameters())
    prior_ids = {id(p) for p in prior_params}
    main_params = [p for p in model.parameters() if id(p) not in prior_ids]
    opt = torch.optim.Adam([
        {"params": main_params, "lr": args.lr},
        {"params": prior_params, "lr": args.lr * 10},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0)
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        agg = {"loss": 0.0, "bpp": 0.0, "mse": 0.0, "n": 0}
        for step, x in enumerate(dl):
            x = x.to(device, non_blocking=True)
            out = model(x)
            r = rate_distortion_loss(out, x, args.lmbda)
            opt.zero_grad(set_to_none=True)
            r["loss"].backward()
            # GDN can produce large gradients early; clipping keeps the first
            # few hundred steps from diverging.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            agg["loss"] += r["loss"].item(); agg["bpp"] += r["bpp"].item()
            agg["mse"] += r["mse"].item(); agg["n"] += 1
            if (step + 1) % args.log_every == 0:
                n = agg["n"]
                print(f"  e{epoch+1} {step+1}/{len(dl)}  loss {agg['loss']/n:8.3f}  "
                      f"bpp {agg['bpp']/n:6.3f}  psnr {psnr(agg['mse']/n):6.2f}dB")

        sched.step()
        n = max(1, agg["n"])
        ep_loss, ep_bpp, ep_psnr = agg["loss"]/n, agg["bpp"]/n, psnr(agg["mse"]/n)
        print(f"epoch {epoch+1}/{args.epochs}  loss {ep_loss:8.3f}  bpp {ep_bpp:6.3f}  "
              f"psnr {ep_psnr:6.2f}dB  ({time.time()-t0:.0f}s)")

        metrics = {"bpp": ep_bpp, "psnr": ep_psnr}
        select = ep_loss
        if val_dl is not None:
            v_loss, v_bpp, v_psnr = evaluate(model, val_dl, args.lmbda, device)
            print(f"           val  loss {v_loss:8.3f}  bpp {v_bpp:6.3f}  psnr {v_psnr:6.2f}dB")
            metrics.update({"val_bpp": v_bpp, "val_psnr": v_psnr, "val_loss": v_loss})
            select = v_loss

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
              "config": model.config, "lmbda": args.lmbda, "metrics": metrics}
        torch.save(ck, args.out)
        if select < best:
            best = select
            torch.save(ck, args.out.replace(".pt", ".best.pt"))
            print(f"           new best ({'val' if val_dl else 'train'} loss {select:.3f})")

    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
