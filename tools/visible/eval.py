import argparse
import io
import random
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import gemini_logo
import kling_logo
import marks

LOGOS = {"gemini": gemini_logo, "kling": kling_logo, "marks": marks}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def reencode(rgb, fmt, **options):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, fmt, **options)
    return gemini_logo.load_rgb(io.BytesIO(buffer.getvalue()))


def rescale(rgb, factor):
    if factor >= 1:
        return rgb
    image = Image.fromarray(rgb)
    return np.asarray(image.resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS))


def crop(rgb, factor):
    height, width, _ = rgb.shape
    h, w = round(height * factor), round(width * factor)
    top, left = (height - h) // 2, (width - w) // 2
    return rgb[top : top + h, left : left + w]


TRANSFORMS = {
    "original": lambda a: a,
    "JPEG 85": lambda a: reencode(a, "JPEG", quality=85),
    "WebP 80": lambda a: reencode(a, "WEBP", quality=80),
    "800 px + JPEG 75": lambda a: reencode(rescale(a, 800 / a.shape[1]), "JPEG", quality=75),
    "50 % + JPEG 80": lambda a: reencode(rescale(a, 0.5), "JPEG", quality=80),
    "33 % + JPEG 80": lambda a: reencode(rescale(a, 1 / 3), "JPEG", quality=80),
    "crop 95 %": lambda a: crop(a, 0.95),
}


def resized(alpha, size):
    return np.asarray(Image.fromarray(alpha.astype(np.float32)).resize((size, size), Image.Resampling.BOX), np.float64)


@cache
def layouts():
    m = gemini_logo.masks()
    return {
        "V1 1024 x 1024": ((1024, 1024), m["bg_48_png"], 32),
        "V1 2816 x 1536": ((2816, 1536), m["bg_96_png"], 64),
        "V2 1024 x 1024": ((1024, 1024), m["bg_b_36_png"], 71),
        "V2 1376 x 768": ((1376, 768), resized(m["bg_b_96_png"], 48), 96),
        "V2 2816 x 1536": ((2816, 1536), m["bg_b_96_png"], 192),
    }


def job(args):
    path, logo, apply, transforms = args
    try:
        rgb = gemini_logo.load_rgb(path)
    except OSError:
        return None
    bases = {"": rgb}
    if apply:
        bases = {
            f"{layout:15s} ": gemini_logo.apply_logo(np.asarray(ImageOps.fit(Image.fromarray(rgb), size, Image.Resampling.LANCZOS)), alpha, margin)
            for layout, (size, alpha, margin) in layouts().items()
        }
    module = LOGOS[logo]
    t = module.template()
    return {prefix + name: module.score(TRANSFORMS[name](base), t) for prefix, base in bases.items() for name in transforms}


def main():
    parser = argparse.ArgumentParser(description="Score image folders with a visible logo check.")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--logo", choices=LOGOS, default="gemini")
    parser.add_argument("--transforms", action="store_true", help="also score re-encoded, rescaled and cropped copies")
    parser.add_argument("--apply", action="store_true", help="overlay the Gemini logo in each Gemini layout before scoring")
    parser.add_argument("--sample", type=int, help="score a random sample of this many images per folder")
    args = parser.parse_args()
    if args.apply and args.logo != "gemini":
        parser.error("--apply needs --logo gemini")
    threshold = LOGOS[args.logo].THRESHOLD

    transforms = list(TRANSFORMS) if args.transforms else ["original"]
    for folder in args.folders:
        paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if args.sample and len(paths) > args.sample:
            paths = sorted(random.Random(1).sample(paths, args.sample))
        with ProcessPoolExecutor() as pool:
            results = [r for r in pool.map(job, [(p, args.logo, args.apply, transforms) for p in paths], chunksize=4) if r]
        print(f"{folder}: {len(results)} images")
        for key in results[0]:
            s = np.array([r[key] for r in results])
            print(
                f"  {key:34s} hit rate {np.mean(s >= threshold) * 100:5.1f} % ({np.sum(s >= threshold)})"
                f"  median {np.median(s):.3f}  p99.9 {np.quantile(s, 0.999):.3f}  max {s.max():.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
