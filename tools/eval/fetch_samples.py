import argparse
import hashlib
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

OUT = Path(__file__).parent / ".cache" / "samples"
OPENFAKE = "datasets/ComplexDataLab/OpenFake"

OPENFAKE_SOURCES = [
    {"group": "core", "config": "core", "quota": {"real": 1000, "fake": 1000}, "per_generator": 40, "max_groups": 13},
    {"group": "reddit", "config": "reddit", "quota": {"real": 500, "fake": 300}, "per_generator": None, "max_groups": 15},
]

PLAIN_SOURCES = [
    {"group": "coco", "source": "coco", "count": 1000, "max_groups": 1,
     "paths": ["datasets/detection-datasets/coco@~parquet/default/val/0000.parquet"]},
    {"group": "charts", "source": "chartqa", "count": 300, "max_groups": 25,
     "paths": ["datasets/HuggingFaceM4/ChartQA@~parquet/default/test/0000.parquet"]},
    {"group": "art", "source": "wikiart", "count": 300, "max_groups": 4,
     "paths": ["datasets/huggan/wikiart@~parquet/default/partial-train/0000.parquet"]},
]


def save(group, label, source, ident, data):
    ext = "png" if data[:4] == b"\x89PNG" else "webp" if data[8:12] == b"WEBP" else "jpg"
    source = re.sub(r"[^A-Za-z0-9._-]", "_", str(source))
    target = OUT / group / label
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{ident}__{source}.{ext}").write_bytes(data)


def row_groups(fs, paths, rng, limit):
    files = [pq.ParquetFile(fs.open(p, block_size=1 << 20)) for p in paths]
    order = [(n, i) for n, f in enumerate(files) for i in range(f.metadata.num_row_groups)]
    rng.shuffle(order)
    for n, i in order[:limit]:
        yield n, i, files[n]


def fetch_openfake(fs, rng, spec):
    paths = sorted(p for p in fs.ls(f"{OPENFAKE}/{spec['config']}", detail=False) if "/test-" in p)
    taken = Counter()
    per_generator = Counter()
    for shard, index, f in row_groups(fs, paths, rng, spec["max_groups"]):
        if all(taken[label] >= q for label, q in spec["quota"].items()):
            break
        rows = f.read_row_group(index, columns=["label", "model", "type"]).to_pylist()
        wanted = []
        for n, row in enumerate(rows):
            label = row["label"]
            if row["type"] == "video" or taken[label] >= spec["quota"][label]:
                continue
            if label == "fake" and spec["per_generator"]:
                if per_generator[row["model"]] >= spec["per_generator"]:
                    continue
                per_generator[row["model"]] += 1
            taken[label] += 1
            wanted.append(n)
        if not wanted:
            continue
        images = f.read_row_group(index, columns=["image"]).column("image").to_pylist()
        for n in wanted:
            save(spec["group"], rows[n]["label"], rows[n]["model"], f"{shard:02d}-{index}-{n}", images[n]["bytes"])
    return dict(taken), len(per_generator)


def fetch_plain(fs, rng, spec):
    seen = set()
    for shard, index, f in row_groups(fs, spec["paths"], rng, spec["max_groups"]):
        images = f.read_row_group(index, columns=["image"]).column("image").to_pylist()
        order = list(range(len(images)))
        rng.shuffle(order)
        for n in order:
            if len(seen) >= spec["count"]:
                return len(seen)
            data = images[n]["bytes"]
            digest = hashlib.sha1(data).digest()
            if digest in seen:
                continue
            seen.add(digest)
            save(spec["group"], "real", spec["source"], f"{shard:02d}-{index}-{n}", data)
    return len(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fs = HfFileSystem()
    shutil.rmtree(OUT, ignore_errors=True)
    for spec in OPENFAKE_SOURCES:
        taken, generators = fetch_openfake(fs, rng, spec)
        print(spec["group"], taken, "generators", generators)
    for spec in PLAIN_SOURCES:
        print(spec["group"], fetch_plain(fs, rng, spec))


if __name__ == "__main__":
    main()
