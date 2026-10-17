"""Fine-tunes Detectra (timm ViT-S/384) on OpenFake train plus negatives with web-compression augmentation."""
import argparse
import glob
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import timm
import timm.optim
import timm.utils
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from fetch_data import DEFAULT_DATA

Image.MAX_IMAGE_PIXELS = None
SHORTEST_EDGE, CROP = 440, 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
RESAMPLES = [Image.LANCZOS, Image.BICUBIC, Image.BILINEAR, Image.BOX]


def create_model(checkpoint=None, drop_path=0.0):
    model = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1, drop_path_rate=drop_path)
    if checkpoint:
        model.load_state_dict({k.removeprefix("vit."): v for k, v in load_file(checkpoint).items()})
    return model


def resize_short(image, short):
    w, h = image.size
    scale = short / min(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


def to_array(image):
    return ((np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def complete(data, pattern):
    return sorted(p for p in glob.glob(str(data / pattern)) if not p.endswith(".part"))


def row_groups(files, label, fraction):
    out = []
    for path in files:
        meta = pq.ParquetFile(path).metadata
        out += [{"path": path, "group": i, "rows": meta.row_group(i).num_rows, "label": label, "fraction": fraction}
                for i in range(meta.num_row_groups)]
    return out


def read_group(g):
    f = pq.ParquetFile(g["path"])
    columns = ["image"] + (["label", "type"] if g["label"] == "column" else [])
    out = []
    for r in f.read_row_group(g["group"], columns=columns).to_pylist():
        if g["label"] == "column":
            if r["type"] == "video":
                continue
            label = 1.0 if r["label"] == "fake" else 0.0
        else:
            label = float(g["label"])
        data = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
        if data:
            out.append((data, label))
    return out


def jpeg(image, quality, subsampling=2):
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def web(image, max_side, quality, resample=Image.LANCZOS, draft=False):
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        if draft and image.format == "JPEG":
            image.draft("RGB", size)
        image = image.convert("RGB").resize(size, resample)
    return jpeg(image.convert("RGB"), quality)


def augment(data, rng):
    image = Image.open(io.BytesIO(data))
    if rng.random() < 0.2:
        image = image.convert("RGB")
    else:
        max_side = int(math.exp(rng.uniform(math.log(560), math.log(2048))))
        image = web(image, max_side, rng.randint(50, 95), rng.choice(RESAMPLES), draft=rng.random() < 0.5)
        if rng.random() < 0.15:
            image = jpeg(image, rng.randint(60, 95), rng.choice([0, 2]))
    image = resize_short(image, rng.randint(400, 480) if rng.random() < 0.5 else SHORTEST_EDGE)
    w, h = image.size
    x, y = rng.randint(0, w - CROP), rng.randint(0, h - CROP)
    image = image.crop((x, y, x + CROP, y + CROP))
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return to_array(image)


VAL_CONDITIONS = {
    "original": lambda im: im.convert("RGB"),
    "web1280": lambda im: web(im, 1280, 80),
    "web800": lambda im: web(im, 800, 75),
}


def val_views(data):
    out = []
    for fn in VAL_CONDITIONS.values():
        image = resize_short(fn(Image.open(io.BytesIO(data))), SHORTEST_EDGE)
        w, h = image.size
        left, top = (w - CROP) // 2, (h - CROP) // 2
        out.append(to_array(image.crop((left, top, left + CROP, top + CROP))))
    return np.stack(out)


class Stream(IterableDataset):
    def __init__(self, groups, seed, buffer=192):
        self.groups, self.seed, self.buffer, self.epoch = groups, seed, buffer, 0

    def epoch_groups(self):
        rng = random.Random(self.seed * 7919 + self.epoch)
        chosen = [g for g in self.groups for _ in range(int(g["fraction"]) + (rng.random() < g["fraction"] % 1))]
        rng.shuffle(chosen)
        return chosen

    def __iter__(self):
        info = get_worker_info()
        wid, workers = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random((self.seed * 7919 + self.epoch) * 64 + wid)
        buffer = []
        for g in self.epoch_groups()[wid::workers]:
            buffer += read_group(g)
            rng.shuffle(buffer)
            while len(buffer) > self.buffer:
                yield from self.sample(buffer.pop(), rng)
        rng.shuffle(buffer)
        for item in buffer:
            yield from self.sample(item, rng)

    @staticmethod
    def sample(item, rng):
        try:
            yield torch.from_numpy(augment(item[0], rng)), torch.tensor(item[1])
        except Exception as e:
            print("skip", type(e).__name__, e, flush=True)


class ValSet(IterableDataset):
    def __init__(self, groups):
        self.groups = groups

    def __iter__(self):
        info = get_worker_info()
        wid, workers = (info.id, info.num_workers) if info else (0, 1)
        for g in self.groups[wid::workers]:
            for data, label in read_group(g):
                try:
                    yield torch.from_numpy(val_views(data)), torch.tensor(label)
                except Exception as e:
                    print("skip val", type(e).__name__, e, flush=True)


def auc(pos, neg):
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(model, loader):
    model.eval()
    logits, labels = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader:
            b, v = x.shape[:2]
            logits.append(model(x.view(b * v, 3, CROP, CROP).cuda(non_blocking=True)).float().view(b, v).cpu().numpy())
            labels.append(y.numpy())
    logits, labels = np.concatenate(logits), np.concatenate(labels)
    result = {}
    for i, name in enumerate(VAL_CONDITIONS):
        pos, neg = logits[labels == 1, i], logits[labels == 0, i]
        threshold = np.quantile(neg, 0.99)
        result[name] = {"auc": round(float(auc(pos, neg)), 4), "tpr@1%": round(float((pos > threshold).mean()), 4)}
    model.train()
    return result


def state(model):
    return {k: v.detach().float().contiguous().cpu() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--newest-weight", type=float, default=3.0)
    ap.add_argument("--max-openfake-files", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    openfake = complete(args.data, "openfake/core/train-*.parquet")
    if args.max_openfake_files:
        openfake = openfake[: args.max_openfake_files]
    val_file, train_files = openfake[-1], openfake[:-1]
    groups = row_groups(train_files, "column", 1.0)
    groups += row_groups(complete(args.data, "openfake-newest/*.parquet"), 1, args.newest_weight)
    groups += row_groups(complete(args.data, "met/*.parquet"), 0, 1.0)
    groups += row_groups(complete(args.data, "coco/*.parquet"), 0, 0.6)
    groups += row_groups(complete(args.data, "chartqa/*.parquet"), 0, 0.3)
    val_groups = row_groups([val_file], "column", 1.0)
    rows = sum(g["rows"] * g["fraction"] for g in groups)
    total = int(rows / args.batch) * args.epochs
    print(f"train files {len(train_files)}, val {Path(val_file).name}, ~{int(rows)} rows/epoch, {total} steps", flush=True)

    out = args.data / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args) | {"train_files": train_files, "val_file": val_file}, indent=1, default=str))

    torch.manual_seed(args.seed)
    model = create_model(args.data / "detectra-v1" / "model.safetensors", args.drop_path).cuda().train()
    ema = timm.utils.ModelEmaV3(model, decay=args.ema)
    opt = timm.optim.create_optimizer_v2(model, "adamw", lr=args.lr, weight_decay=0.05, layer_decay=args.layer_decay)
    warmup = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup)))))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    stream = Stream(groups, args.seed)
    val_loader = DataLoader(ValSet(val_groups), batch_size=16, num_workers=args.workers, pin_memory=True)
    history = [{"epoch": 0, "val": evaluate(model, val_loader)}]
    print(json.dumps(history[-1]), flush=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        stream.epoch = epoch
        loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
        t0, seen, running = time.time(), 0, 0.0
        for x, y in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x.cuda(non_blocking=True)).float().squeeze(1), y.cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            seen += len(y)
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
