"""Downloads the fine-tuning data: Detectra weights, 16 OpenFake train shards and photos, artworks and charts as negatives."""
import argparse
import io
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main_checkout():
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        return Path(__file__).parents[2]


DEFAULT_DATA = main_checkout() / "tools" / "train" / ".cache"

DETECTRA = ("TensorFold/detectra-v1", "2bbc31161daa46932e05ee6f96e7016ee817bbde", "model.safetensors")
OPENFAKE = ("ComplexDataLab/OpenFake", "3247b942f18d39d1fd476b1077852af033d35d8c",
            [f"core/train-{n:05d}-of-00032-00000.parquet" for n in range(16)])
NEGATIVES = {
    "coco": ("detection-datasets/coco", "26ddc382fe75dfc2a0655b5977e296ea10efebce",
             ["default/train/0000.parquet", "default/train/0001.parquet"]),
    "chartqa": ("HuggingFaceM4/ChartQA", "c50f07aee7370c2d71640adce1d9e095ed259c7d", ["default/train/0000.parquet"]),
}


MET = ("metmuseum/openaccess", "c65f8d6041aea7b3bc767a54d93772c3c6a365f6")
MET_QUOTAS = {"painting": (3500, 200), "drawing": (500, 50), "print": (500, 50)}
MET_COLUMNS = ["objectID", "isPublicDomain", "primaryImage", "classification", "objectName"]
MET_MAX_SIDE = 2048


def download(repo, revision, filename, target, repo_type="dataset"):
    return Path(hf_hub_download(repo, filename, repo_type=repo_type, revision=revision, local_dir=target))


def split_row_groups(source, target):
    if target.exists():
        return
    f = pq.ParquetFile(source)
    part = target.with_suffix(".part")
    writer = None
    for i in range(f.metadata.num_row_groups):
        table = f.read_row_group(i, columns=["image"])
        writer = writer or pq.ParquetWriter(part, table.schema)
        writer.write_table(table, row_group_size=64)
    writer.close()
    part.rename(target)


def met_kind(row):
    if not (row["isPublicDomain"] and row["primaryImage"]):
        return None
    classification = row["classification"] or ""
    if "painting" in classification.lower() or (row["objectName"] or "").lower().startswith("painting"):
        return "painting"
    return {"Drawings": "drawing", "Prints": "print"}.get(classification)


def met_parquet(fs, path):
    return pq.ParquetFile(fs.open(path, block_size=2**18), pre_buffer=True)


def scan_met(fs, path):
    f = met_parquet(fs, path)
    meta = f.metadata
    image = [j for j in range(meta.num_columns) if meta.row_group(0).column(j).path_in_schema.startswith("image")]
    groups = []
    for g in range(meta.num_row_groups):
        kinds = [met_kind(r) for r in f.read_row_group(g, columns=MET_COLUMNS).to_pylist()]
        groups.append({"path": path, "group": g, "kinds": kinds,
                       "bytes": sum(meta.row_group(g).column(j).total_compressed_size for j in image)})
    return groups


def select_met(groups):
    picks = defaultdict(list)
    for kind, (quota, cap) in MET_QUOTAS.items():
        ranked = sorted(groups, key=lambda g: (-g["kinds"].count(kind) / g["bytes"], g["path"], g["group"]))
        taken = 0
        for g in ranked:
            rows = [i for i, k in enumerate(g["kinds"]) if k == kind][: min(cap, quota - taken)]
            if not rows:
                break
            picks[(g["path"], g["group"])] += [(i, kind) for i in rows]
            taken += len(rows)
    return picks


def shrink(data):
    image = Image.open(io.BytesIO(data))
    if max(image.size) <= MET_MAX_SIDE:
        return data
    image.thumbnail((MET_MAX_SIDE, MET_MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


def read_met(fs, path, group, rows):
    column = met_parquet(fs, path).read_row_group(group, columns=["image"]).column("image").to_pylist()
    return [(kind, shrink(column[i]["bytes"])) for i, kind in rows if column[i]]


def fetch_met(target):
    if all((target / f"met-{kind}.parquet").exists() for kind in MET_QUOTAS):
        return
    repo, revision = MET
    fs = HfFileSystem()
    scan = target / "scan.json"
    if not scan.exists():
        files = sorted(fs.glob(f"datasets/{repo}@{revision}/default/train/openaccess-*.parquet"))
        with ThreadPoolExecutor(16) as pool:
            scan.write_text(json.dumps([g for groups in pool.map(lambda p: scan_met(fs, p), files) for g in groups]))
    picks = select_met(json.loads(scan.read_text()))
    keys = sorted(picks)
    images = defaultdict(list)
    with ThreadPoolExecutor(4) as pool:
        for (path, group), rows in zip(keys, pool.map(lambda key: read_met(fs, *key, picks[key]), keys)):
            for kind, data in rows:
                images[kind].append(data)
            print("met", Path(path).name, group, len(rows), flush=True)
    for kind, data in images.items():
        pq.write_table(pa.table({"image": pa.array(data, pa.binary())}), target / f"met-{kind}.parquet", row_group_size=64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()

    repo, revision, filename = DETECTRA
    download(repo, revision, filename, args.data / "detectra-v1", repo_type="model")
    repo, revision, files = OPENFAKE
    for filename in files:
        print("openfake", download(repo, revision, filename, args.data / "openfake"), flush=True)
    (args.data / "met").mkdir(parents=True, exist_ok=True)
    fetch_met(args.data / "met")
    for name, (repo, revision, files) in NEGATIVES.items():
        (args.data / name).mkdir(parents=True, exist_ok=True)
        for filename in files:
            source = download(repo, revision, filename, args.data / "raw" / name)
            target = args.data / name / Path(filename).name
            split_row_groups(source, target)
            print(name, target, pq.ParquetFile(target).metadata.num_rows, flush=True)


if __name__ == "__main__":
    main()
