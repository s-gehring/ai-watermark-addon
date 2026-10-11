import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

import synthid

ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def reencode(rgb, fmt, **options):
    buffer = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buffer, fmt, **options)
    return synthid.load_rgb(io.BytesIO(buffer.getvalue()))


def rescale(rgb, factor):
    image = Image.fromarray(rgb.astype(np.uint8))
    size = (round(image.width * factor), round(image.height * factor))
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), np.float64)


def crop(rgb, factor):
    height, width, _ = rgb.shape
    h, w = round(height * factor), round(width * factor)
    top, left = (height - h) // 2, (width - w) // 2
    return rgb[top : top + h, left : left + w]


TRANSFORMS = {
    "original": lambda a: a,
    "scale 50 %": lambda a: rescale(a, 0.5),
    "scale 33 %": lambda a: rescale(a, 1 / 3),
    "JPEG 90 (4:4:4)": lambda a: reencode(a, "JPEG", quality=90, subsampling=0),
    "JPEG 90 (4:2:0)": lambda a: reencode(a, "JPEG", quality=90, subsampling=2),
    "WebP 80": lambda a: reencode(a, "WEBP", quality=80),
    "crop 95 %": lambda a: crop(a, 0.95),
}


def images_by_format(folder):
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    groups = {"PNG": [p for p in paths if p.suffix.lower() == ".png"], "JPEG/WebP": [p for p in paths if p.suffix.lower() != ".png"]}
    return {name: group for name, group in groups.items() if group}


def main():
    parser = argparse.ArgumentParser(description="Score image folders with the bundled SynthID template.")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--transforms", action="store_true", help="also score re-encoded, rescaled and cropped copies")
    args = parser.parse_args()

    weights, phases, meta = synthid.load(ROOT / "src" / "background" / "synthid-template.bin", ROOT / "synthid.json")
    threshold = meta["threshold"]
    transforms = TRANSFORMS if args.transforms else {"original": TRANSFORMS["original"]}
    for folder in args.folders:
        for fmt, paths in images_by_format(folder).items():
            for name, transform in transforms.items():
                scores = np.array([synthid.score(transform(synthid.load_rgb(p)), weights, phases) for p in paths])
                print(
                    f"{folder:40s} {fmt:9s} {name:16s} n={len(scores):4d}  hit rate {np.mean(scores >= threshold) * 100:5.1f} %"
                    f"  median z {np.median(scores):6.1f}  max z {scores.max():6.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
