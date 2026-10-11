import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

OUT = Path(__file__).parent / ".cache" / "openfake"
REPO = "datasets/ComplexDataLab/OpenFake"
TARGET = "nano-banana-pro"
OTHER_CAP = 40
REAL_CAP = 700
REAL_SHARDS = (0, 3, 5, 9, 11)
REAL_GROUPS_PER_SHARD = 3


def shard_file(fs, index):
    return pq.ParquetFile(fs.open(f"{REPO}/core/test-{index:05d}-of-00013.parquet", block_size=4 << 20))


def groups(f, column):
    values = f.read(columns=[column]).column(column).to_pylist()
    offset = 0
    for g in range(f.metadata.num_row_groups):
        n = f.metadata.row_group(g).num_rows
        yield g, values[offset : offset + n]
        offset += n


def save(row, folder, name):
    data = row["image"]["bytes"]
    ext = "png" if data[:4] == b"\x89PNG" else "webp" if data[8:12] == b"WEBP" else "jpg"
    target = OUT / folder / re.sub(r"[^A-Za-z0-9._-]", "_", row["model"])
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.{ext}").write_bytes(data)


def main():
    fs = HfFileSystem()
    taken = Counter()
    for i in range(13):
        f = shard_file(fs, i)
        for g, models in groups(f, "model"):
            if TARGET not in models:
                continue
            for k, row in enumerate(f.read_row_group(g, columns=["image", "label", "model", "type"]).to_pylist()):
                if row["type"] != "image" or row["label"] != "fake":
                    continue
                if row["model"] != TARGET and taken[row["model"]] >= OTHER_CAP:
                    continue
                taken[row["model"]] += 1
                save(row, "fake", f"{i:02d}-{g}-{k}")
        print(f"shard {i}: {taken[TARGET]} {TARGET} images", flush=True)

    for i in REAL_SHARDS:
        f = shard_file(fs, i)
        ranked = sorted(((m.count("docci") + m.count("imagenet"), g) for g, m in groups(f, "model")), reverse=True)
        for _, g in ranked[:REAL_GROUPS_PER_SHARD]:
            for k, row in enumerate(f.read_row_group(g, columns=["image", "label", "model", "type"]).to_pylist()):
                if row["label"] == "real" and taken[row["model"]] < REAL_CAP:
                    taken[row["model"]] += 1
                    save(row, "real", f"{i:02d}-{g}-{k}")
        print(f"shard {i}: {taken['docci']} docci, {taken['imagenet']} imagenet", flush=True)
    print(dict(taken))


if __name__ == "__main__":
    main()
