import argparse
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

OUT = Path(__file__).parent / ".cache" / "samples"
REPO = "datasets/ComplexDataLab/OpenFake"


def row_groups(fs, shard, count, rng):
    f = pq.ParquetFile(fs.open(f"{REPO}/{shard}", block_size=1 << 20))
    for index in rng.sample(range(f.metadata.num_row_groups), count):
        yield index, f.read_row_group(index, columns=["image", "label", "model", "type"]).to_pylist()


def save(row, set_name, ident):
    data = row["image"]["bytes"]
    ext = "png" if data[:4] == b"\x89PNG" else "webp" if data[8:12] == b"WEBP" else "jpg"
    model = re.sub(r"[^A-Za-z0-9._-]", "_", str(row["model"]))
    target = OUT / set_name / row["label"]
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{ident}__{model}.{ext}").write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--wild", type=int, default=50)
    parser.add_argument("--per-generator", type=int, default=8)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fs = HfFileSystem()
    taken = Counter()
    per_generator = Counter()

    for shard in rng.sample([f"core/test-{i:05d}-of-00013.parquet" for i in range(13)], args.shards):
        for group, rows in row_groups(fs, shard, 1, rng):
            for n, row in enumerate(rows):
                label = row["label"]
                if row["type"] == "video" or taken[label] >= args.per_class:
                    continue
                if label == "fake":
                    if per_generator[row["model"]] >= args.per_generator:
                        continue
                    per_generator[row["model"]] += 1
                taken[label] += 1
                save(row, "core", f"{shard[10:15]}-{group}-{n}")

    wild = 0
    shard = f"reddit/test-{rng.randrange(5):05d}-of-00005.parquet"
    for group, rows in row_groups(fs, shard, 1, rng):
        for n, row in enumerate(rows):
            if wild >= args.wild or row["type"] != "image" or row["label"] != "fake":
                continue
            wild += 1
            save(row, "wild", f"r{group}-{n}")

    print(dict(taken), "wild", wild, "generators", len(per_generator))


if __name__ == "__main__":
    main()
