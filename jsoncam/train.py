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
import copy
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .data import ImageCropDataset, PatchDataset
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


def parse_deadline(text):
    """`2026-09-12T14:00` in local time -> epoch seconds, or None."""
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    raise SystemExit(f"cannot parse --deadline {text!r}; want YYYY-MM-DDTHH:MM")


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


class EMA:
    """Exponential moving average of the weights.

    The raw weights at any one step carry the noise of the last few batches;
    the average over the last couple of thousand steps is usually a little
    better than any single one of them, and it costs one extra copy.
    """

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def main(argv=None):
    ap = argparse.ArgumentParser("jsoncam-train")
    ap.add_argument("--cache", default="data/patches.npy",
                    help="fixed patch cache from `jsoncam prepare`")
    ap.add_argument("--image-cache", default=None,
                    help="multi-scale image cache from `jsoncam prepare-images`; "
                         "crops are cut fresh every step (overrides --cache)")
    ap.add_argument("--patch", type=int, default=256, help="crop size for --image-cache")
    ap.add_argument("--steps-per-epoch", type=int, default=2000,
                    help="with --image-cache the data has no end, so this is one epoch")
    ap.add_argument("--val-cache", default=None,
                    help="held-out patches; best checkpoint is chosen on THIS loss")
    ap.add_argument("--out", default="checkpoints/jc.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--deadline", default=None,
                    help="local time YYYY-MM-DDTHH:MM to stop at; the learning-rate "
                         "schedule reaches its floor by then even if epochs remain")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500, help="linear warm-up steps")
    ap.add_argument("--ema", type=float, default=0.9995, help="EMA decay; 0 disables")
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
          f"batch={args.batch} lr={args.lr} epochs={args.epochs} out={args.out} "
          f"patch={args.patch} warmup={args.warmup} ema={args.ema} "
          f"deadline={args.deadline}")

    if args.image_cache:
        ds = ImageCropDataset(args.image_cache, patch=args.patch,
                              length=args.steps_per_epoch * args.batch)
        print(f"{ds.n_images} images at {len(ds.tiers)} scales, fresh crops every step")
    else:
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
    print(f"{sum(p.numel() for p in model.parameters())} parameters")
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
    base_lrs = [g["lr"] for g in opt.param_groups]
    ema = EMA(model, args.ema) if args.ema > 0 else None

    resumed = None
    if args.resume and os.path.exists(args.resume):
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        opt.load_state_dict(resumed["opt"])
        start_epoch = resumed.get("epoch", 0)
        if ema is not None and resumed.get("ema"):
            ema.model.load_state_dict(resumed["ema"])
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = float("inf")

    # Cosine schedule over whichever is further along: the step count or the
    # wall clock. With a deadline the run is guaranteed to have annealed by
    # the time it is cut off, instead of stopping at some high-LR midpoint.
    total_steps = args.epochs * len(dl)
    t_start = time.time()
    step_global = 0
    if resumed:
        # Carry the clock and the step count across a restart, so the schedule
        # picks up where it was instead of warming up again at full rate.
        t_start = resumed.get("t_start", t_start)
        step_global = resumed.get("step", start_epoch * len(dl))
    deadline = parse_deadline(args.deadline)
    if deadline:
        print(f"deadline: {time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline))} "
              f"({(deadline - t_start)/3600:.1f}h from now)")

    def set_lr():
        if step_global < args.warmup:
            f = (step_global + 1) / args.warmup
        else:
            prog = (step_global - args.warmup) / max(1, total_steps - args.warmup)
            if deadline:
                prog = max(prog, (time.time() - t_start) / max(1.0, deadline - t_start))
            prog = min(1.0, prog)
            f = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        return f

    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        agg = {"loss": 0.0, "bpp": 0.0, "mse": 0.0, "n": 0}
        for step, x in enumerate(dl):
            lr_f = set_lr()
            x = x.to(device, non_blocking=True)
            out = model(x)
            r = rate_distortion_loss(out, x, args.lmbda)
            opt.zero_grad(set_to_none=True)
            r["loss"].backward()
            # GDN can produce large gradients early; clipping keeps the first
            # few hundred steps from diverging.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)
            step_global += 1

            agg["loss"] += r["loss"].item(); agg["bpp"] += r["bpp"].item()
            agg["mse"] += r["mse"].item(); agg["n"] += 1
            if (step + 1) % args.log_every == 0:
                n = agg["n"]
                print(f"  e{epoch+1} {step+1}/{len(dl)}  loss {agg['loss']/n:8.3f}  "
                      f"bpp {agg['bpp']/n:6.3f}  psnr {psnr(agg['mse']/n):6.2f}dB  "
                      f"lr x{lr_f:.3f}", flush=True)
            if deadline and time.time() > deadline:
                print(f"  deadline reached at step {step+1}/{len(dl)}; finishing epoch early")
                stop = True
                break

        n = max(1, agg["n"])
        ep_loss, ep_bpp, ep_psnr = agg["loss"]/n, agg["bpp"]/n, psnr(agg["mse"]/n)
        print(f"epoch {epoch+1}/{args.epochs}  loss {ep_loss:8.3f}  bpp {ep_bpp:6.3f}  "
              f"psnr {ep_psnr:6.2f}dB  ({time.time()-t0:.0f}s)")

        metrics = {"bpp": ep_bpp, "psnr": ep_psnr}
        select, weights, which = ep_loss, model.state_dict(), "raw"
        if val_dl is not None:
            v_loss, v_bpp, v_psnr = evaluate(model, val_dl, args.lmbda, device)
            print(f"           val  loss {v_loss:8.3f}  bpp {v_bpp:6.3f}  psnr {v_psnr:6.2f}dB")
            metrics.update({"val_bpp": v_bpp, "val_psnr": v_psnr, "val_loss": v_loss})
            select = v_loss
            if ema is not None:
                e_loss, e_bpp, e_psnr = evaluate(ema.model, val_dl, args.lmbda, device)
                print(f"           ema  loss {e_loss:8.3f}  bpp {e_bpp:6.3f}  psnr {e_psnr:6.2f}dB")
                metrics.update({"ema_bpp": e_bpp, "ema_psnr": e_psnr, "ema_loss": e_loss})
                if e_loss < v_loss:
                    select, weights, which = e_loss, ema.model.state_dict(), "ema"

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
              "ema": ema.model.state_dict() if ema is not None else None,
              "config": model.config, "lmbda": args.lmbda, "metrics": metrics,
              "t_start": t_start, "step": step_global}
        torch.save(ck, args.out)
        if select < best:
            best = select
            # The best file carries whichever weights won under "model", so
            # `export` and `load_checkpoint` need no idea EMA exists.
            torch.save({**ck, "model": weights, "weights": which},
                       args.out.replace(".pt", ".best.pt"))
            print(f"           new best ({'val' if val_dl else 'train'} loss {select:.3f}) [{which}]",
                  flush=True)
        if stop:
            break

    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
