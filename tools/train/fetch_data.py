"""Downloads the fine-tuning data: Detectra weights, 16 OpenFake train shards and photos, paintings and charts as negatives."""
import argparse
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


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
    "wikiart": ("huggan/wikiart", "d559852d2b232e0fcf195e775866964f0564f2b5",
                [f"data/train-{n:05d}-of-00072.parquet" for n in range(60, 64)]),
    "coco": ("detection-datasets/coco", "26ddc382fe75dfc2a0655b5977e296ea10efebce",
             ["default/train/0000.parquet", "default/train/0001.parquet"]),
    "chartqa": ("HuggingFaceM4/ChartQA", "c50f07aee7370c2d71640adce1d9e095ed259c7d", ["default/train/0000.parquet"]),
}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()

    repo, revision, filename = DETECTRA
    download(repo, revision, filename, args.data / "detectra-v1", repo_type="model")
    repo, revision, files = OPENFAKE
    for filename in files:
        print("openfake", download(repo, revision, filename, args.data / "openfake"), flush=True)
    for name, (repo, revision, files) in NEGATIVES.items():
        (args.data / name).mkdir(parents=True, exist_ok=True)
        for filename in files:
            source = download(repo, revision, filename, args.data / "raw" / name)
            target = args.data / name / Path(filename).name
            split_row_groups(source, target)
            print(name, target, pq.ParquetFile(target).metadata.num_rows, flush=True)


if __name__ == "__main__":
    main()
