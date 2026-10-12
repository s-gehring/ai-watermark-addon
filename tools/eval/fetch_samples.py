import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, snapshot_download
from PIL import Image


def main_checkout():
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        return Path(__file__).parents[2]


CACHE = main_checkout() / "tools" / "eval" / ".cache"
OUT = CACHE / "samples"
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


RESEARCH = {
    "repo": "bigdatamark/synthid-research",
    "revision": "af707a1b8823b277f260a1ebfbae3c9877d3d897",
    "folders": ["gemini-3.1-flash-image", "qwen-image-2.1", "flux.2", "gpt-image-2"],
    "per_folder": 150,
}

MUSEUM = {
    "api": "https://openaccess-api.clevelandart.org/api/artworks/",
    "type": "Painting",
    "count": 300,
}
USER_AGENT = {"User-Agent": "ai-watermark-addon-eval"}

YOUTUBE = {
    "segments": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
    "music": "/m/04rlf",
    "per_kind": 300,
}


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


def fetch_research(rng, spec):
    files = sorted(HfApi().list_repo_files(spec["repo"], repo_type="dataset", revision=spec["revision"]))
    chosen = []
    for folder in spec["folders"]:
        candidates = [f for f in files if f.startswith(f"{folder}/") and not f.endswith((".json", ".csv", ".txt"))]
        chosen += rng.sample(candidates, min(spec["per_folder"], len(candidates)))
    local = CACHE / "downloads" / spec["repo"].split("/")[1]
    snapshot_download(spec["repo"], repo_type="dataset", revision=spec["revision"], local_dir=local, allow_patterns=chosen, max_workers=8)
    target = OUT / "research" / "fake"
    target.mkdir(parents=True, exist_ok=True)
    for path in chosen:
        folder, name = path.split("/", 1)
        os.link(local / path, target / f"{Path(name).stem}__{folder}{Path(name).suffix}")
    return Counter(path.split("/", 1)[0] for path in chosen)


def fetch_museum(rng, spec):
    items, skip = [], 0
    while True:
        query = urllib.parse.urlencode({"cc0": 1, "has_image": 1, "type": spec["type"], "limit": 1000, "skip": skip, "fields": "id,images"})
        with urllib.request.urlopen(urllib.request.Request(f"{spec['api']}?{query}", headers=USER_AGENT), timeout=60) as res:
            data = json.load(res)["data"]
        items += [(r["id"], r["images"]["web"]["url"]) for r in data if (r.get("images") or {}).get("web")]
        if len(data) < 1000:
            break
        skip += 1000
    for ident, url in rng.sample(sorted(items), spec["count"]):
        with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=60) as res:
            save("art", "real", "cma", ident, res.read())
    return spec["count"]


def youtube_thumbnail(ytid):
    try:
        with urllib.request.urlopen(f"https://i.ytimg.com/vi/{ytid}/hqdefault.jpg", timeout=20) as res:
            image = Image.open(BytesIO(res.read())).convert("RGB")
    except (urllib.error.URLError, OSError):
        return None
    height = round(image.width * 9 / 16)
    top = (image.height - height) // 2
    out = BytesIO()
    image.crop((0, top, image.width, top + height)).save(out, "JPEG", quality=90)
    return out.getvalue()


def fetch_youtube(rng, spec):
    path = CACHE / "downloads" / "audioset" / "eval_segments.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(spec["segments"], path)
    rows = [line.split(", ", 3) for line in path.read_text().splitlines() if not line.startswith("#")]
    kinds = {
        "music": [r[0] for r in rows if spec["music"] in r[3]],
        "other": [r[0] for r in rows if spec["music"] not in r[3]],
    }
    taken = Counter()
    for kind, ids in kinds.items():
        rng.shuffle(ids)
        for ytid in ids:
            if taken[kind] >= spec["per_kind"]:
                break
            data = youtube_thumbnail(ytid)
            if data:
                save("youtube", "real", f"audioset-{kind}", ytid, data)
                taken[kind] += 1
    return dict(taken)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-youtube", action="store_true")
    parser.add_argument("--only-museum", action="store_true")
    args = parser.parse_args()

    if args.only_museum:
        for path in (OUT / "art" / "real").glob("*__cma.*"):
            path.unlink()
        print("museum", fetch_museum(random.Random(f"{args.seed}-museum"), MUSEUM))
        return

    if args.only_youtube:
        shutil.rmtree(OUT / "youtube", ignore_errors=True)
        print("youtube", fetch_youtube(random.Random(f"{args.seed}-youtube"), YOUTUBE))
        return
    rng = random.Random(args.seed)
    fs = HfFileSystem()
    shutil.rmtree(OUT, ignore_errors=True)
    for spec in OPENFAKE_SOURCES:
        taken, generators = fetch_openfake(fs, rng, spec)
        print(spec["group"], taken, "generators", generators)
    for spec in PLAIN_SOURCES:
        print(spec["group"], fetch_plain(fs, rng, spec))
    print("research", dict(fetch_research(random.Random(f"{args.seed}-research"), RESEARCH)))
    print("museum", fetch_museum(random.Random(f"{args.seed}-museum"), MUSEUM))
    print("youtube", fetch_youtube(random.Random(f"{args.seed}-youtube"), YOUTUBE))


if __name__ == "__main__":
    main()
