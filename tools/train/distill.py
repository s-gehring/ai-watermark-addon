"""Distils two fine-tuned detectors into one: a photo model (trained with modern photos and hard negatives) and an art model (trained without).
Per image, the teacher mixes their Platt-calibrated logits by the CLIP router (route_clip.py): (1 - q) * photo + q * (art + offset) with q = p(art) ** power.
The student starts from the art model and learns the mix as a soft label, mapped back into the art model's raw logit space."""
import argparse
import json
import math
import time
from pathlib import Path

import timm.optim
import timm.utils
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from fetch_data import DEFAULT_DATA
from route_clip import key
from train import Stream, ValSet, augment, complete, create_model, evaluate, row_groups, state


class RoutedStream(Stream):
    def __init__(self, groups, seed, router):
        super().__init__(groups, seed)
        self.router = router

    def sample(self, item, rng):
        try:
            yield torch.from_numpy(augment(item[0], rng)), torch.tensor(self.router[key(item[0])])
        except Exception as e:
            print("skip", type(e).__name__, e, flush=True)


def platt(text):
    weight, bias = map(float, text.split(","))
    return weight, bias


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photo_teacher", type=Path)
    ap.add_argument("art_teacher", type=Path)
    ap.add_argument("--photo-platt", type=platt, required=True, help="WEIGHT,BIAS")
    ap.add_argument("--art-platt", type=platt, required=True, help="WEIGHT,BIAS")
    ap.add_argument("--art-offset", type=float, default=-1.5)
    ap.add_argument("--router-power", type=float, default=0.5)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--newest-weight", type=float, default=2.0)
    ap.add_argument("--selfgen-weight", type=float, default=1.0)
    ap.add_argument("--modern-photo-weight", type=float, default=0.5)
    ap.add_argument("--hard-negative-weight", type=float, default=2.0)
    ap.add_argument("--art-weight", type=float, default=1.0)
    ap.add_argument("--max-openfake-files", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    openfake = complete(args.data, "openfake/core/train-*.parquet")[: args.max_openfake_files or None]
    val_file, train_files = openfake[-1], openfake[:-1]
    groups = row_groups(train_files, "column", 1.0)
    groups += row_groups(complete(args.data, "openfake-newest/*.parquet"), 1, args.newest_weight)
    groups += row_groups(complete(args.data, "openfake-photos/*.parquet"), 0, args.newest_weight)
    groups += row_groups(complete(args.data, "selfgen/*.parquet"), 1, args.selfgen_weight)
    groups += row_groups(complete(args.data, "modern-photos/*.parquet"), 0, args.modern_photo_weight)
    groups += row_groups(complete(args.data, "hard-negatives/*.parquet"), 0, args.hard_negative_weight)
    groups += row_groups(complete(args.data, "met/*.parquet"), 0, 1.0)
    groups += row_groups(complete(args.data, "art-real/*.parquet"), 0, args.art_weight)
    groups += row_groups(complete(args.data, "art-fake/*.parquet"), 1, args.art_weight)
    groups += row_groups(complete(args.data, "coco/*.parquet"), 0, 0.6)
    groups += row_groups(complete(args.data, "chartqa/*.parquet"), 0, 0.3)
    rows = sum(g["rows"] * g["fraction"] for g in groups)
    total = int(rows / args.batch) * args.epochs
    print(f"train files {len(train_files)}, val {Path(val_file).name}, ~{int(rows)} rows/epoch, {total} steps", flush=True)

    out = args.data / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args) | {"train_files": train_files, "val_file": val_file}, indent=1, default=str))

    router = json.loads((args.data / "router.json").read_text())
    (pa, pb), (aa, ab) = args.photo_platt, args.art_platt
    teachers = [create_model(p).cuda().eval().to(torch.bfloat16).requires_grad_(False) for p in (args.photo_teacher, args.art_teacher)]

    torch.manual_seed(args.seed)
    model = create_model(args.art_teacher, args.drop_path).cuda().train()
    ema = timm.utils.ModelEmaV3(model, decay=args.ema)
    opt = timm.optim.create_optimizer_v2(model, "adamw", lr=args.lr, weight_decay=0.05, layer_decay=args.layer_decay)
    warmup = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup)))))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    stream = RoutedStream(groups, args.seed, router)
    val_loader = DataLoader(ValSet(row_groups([val_file], "column", 1.0)), batch_size=16, num_workers=args.workers, pin_memory=True)
    history = [{"epoch": 0, "val": evaluate(model, val_loader)}]
    print(json.dumps(history[-1]), flush=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        stream.epoch = epoch
        loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
        t0, seen, running = time.time(), 0, 0.0
        for x, p in loader:
            x, q = x.cuda(non_blocking=True), p.cuda(non_blocking=True) ** args.router_power
            with torch.no_grad():
                zp, za = (t(x.to(torch.bfloat16)).float().squeeze(1) for t in teachers)
                mix = (1 - q) * (pa * zp + pb) + q * (aa * za + ab + args.art_offset)
                target = torch.sigmoid((mix - ab) / aa)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x).float().squeeze(1), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            seen += len(x)
            running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {running:.4f} lr {sched.get_last_lr()[0]:.2e} {seen / (time.time() - t0):.0f} img/s", flush=True)
        history.append({"epoch": epoch, "steps": step, "val": evaluate(model, val_loader), "val_ema": evaluate(ema.module, val_loader)})
        print(json.dumps(history[-1]), flush=True)
        save_file(state(model), str(out / f"epoch{epoch}.safetensors"))
        save_file(state(ema.module), str(out / f"epoch{epoch}-ema.safetensors"))
        (out / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
