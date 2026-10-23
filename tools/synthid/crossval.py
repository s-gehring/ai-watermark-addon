import argparse
import json
from multiprocessing import Pool

import numpy as np

import synthid
from build_templates import (
    AMAZON_BORDER,
    AMAZON_MIN_CHANNEL_STD,
    AMAZON_MIN_FREQUENCY,
    AMAZON_SIZE,
    CR,
    GOOGLE_SIZE,
    META,
    OPENAI_BORDER,
    OPENAI_MAX_FREQUENCY,
    OPENAI_SIZES,
    amazon_images,
    google_images,
    openai_images,
    survival,
)
from eval import TRANSFORMS


def rates(z, thresholds):
    return "  ".join(f"z >= {t:g}: {np.mean(z >= t) * 100:5.1f} %" for t in thresholds)


def google_scores(args):
    path, templates, transforms = args
    rgb = synthid.load_rgb(path)
    scores = {}
    for name in transforms:
        image = TRANSFORMS[name](rgb)
        u = synthid.spectrum(image, GOOGLE_SIZE)[0]
        height, width, _ = image.shape
        scores[name] = [synthid.score_spectrum(u, height, width, w, p) for w, p in templates]
    return scores


def google(transforms, thresholds):
    shipped = [t for t in json.loads(META.read_text())["templates"] if t["provider"] == "google"]
    images = google_images()
    paths = sorted(p for group in images.values() for p in group)
    dataset = {p: d for d, group in images.items() for p in group}
    results = []
    for fold in range(3):
        train, test = [p for i, p in enumerate(paths) if i % 3 != fold], [p for i, p in enumerate(paths) if i % 3 == fold]
        weights, phases, _ = synthid.build(train, GOOGLE_SIZE)
        templates = [synthid.dequantize(*synthid.quantize(w, phases)) for w in (weights, weights * survival(train, GOOGLE_SIZE))]
        with Pool() as pool:
            for path, scores in zip(test, pool.imap(google_scores, [(p, templates, list(transforms)) for p in test])):
                results.append((dataset[path], scores))
        print(f"google fold {fold}: train {len(train)}, test {len(test)}", flush=True)
    for source in images:
        for name in transforms:
            z = np.array([scores[name] for d, scores in results if d == source])
            hit = np.any(z >= [t["threshold"] for t in shipped], axis=1)
            print(
                f"{source:32s} {name:16s} n={len(z):3d}  median z {np.median(z[:, 0]):5.1f} / {np.median(z[:, 1]):5.1f}  detected {hit.mean() * 100:5.1f} %  "
                f"{rates(z.max(axis=1), thresholds)}",
                flush=True,
            )


def openai(transforms, thresholds):
    for (width, height), paths_by_dataset in openai_images().items():
        paths = sorted(p for group in paths_by_dataset.values() for p in group)
        train, test = [p for i, p in enumerate(paths) if i % 3 != 2], [p for i, p in enumerate(paths) if i % 3 == 2]
        size = OPENAI_SIZES[(width, height)]
        weights, phases, _ = synthid.build(train, size, OPENAI_MAX_FREQUENCY, OPENAI_BORDER)
        weights, phases = synthid.dequantize(*synthid.quantize(weights, phases))
        for name, transform in transforms.items():
            z = np.array([synthid.score(transform(synthid.load_rgb(p)), weights, phases, size) for p in test])
            print(f"{width}x{height}  train {len(train):3d}  {name:16s} n={len(z):3d}  median z {np.median(z):5.1f}  {rates(z, thresholds)}", flush=True)


def amazon(transforms, thresholds):
    groups = amazon_images()
    z = {name: [] for name in transforms}
    for held, test in groups.items():
        train = [p for repository, paths in groups.items() if repository != held for p in paths]
        weights, phases, _ = synthid.build(
            train, AMAZON_SIZE, min_frequency=AMAZON_MIN_FREQUENCY, border=AMAZON_BORDER, channels=(CR,)
        )
        weights, phases = synthid.dequantize(*synthid.quantize(weights, phases))
        for name, transform in transforms.items():
            scores = [
                synthid.score(transform(synthid.load_rgb(p)), weights, phases, AMAZON_SIZE, AMAZON_MIN_CHANNEL_STD) for p in test
            ]
            z[name] += scores
        print(f"nova-canvas  held out {held:50s} n={len(test):3d}  median z {np.median(z['original'][-len(test):]):5.1f}", flush=True)
    for name, scores in z.items():
        scores = np.array(scores)
        print(f"nova-canvas  {name:16s} n={len(scores):3d}  median z {np.median(scores):5.1f}  {rates(scores, thresholds)}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Score held-out images: Google and OpenAI templates from two thirds of their images, the Amazon template leaving out one repository at a time."
    )
    parser.add_argument("providers", nargs="*", choices=["google", "openai", "amazon"], default=["google", "openai", "amazon"])
    parser.add_argument("--transforms", action="store_true", help="also score re-encoded, rescaled and cropped copies")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[8, 9, 10, 12])
    args = parser.parse_args()

    transforms = TRANSFORMS if args.transforms else {"original": TRANSFORMS["original"]}
    if "google" in args.providers:
        google(transforms, args.thresholds)
    if "openai" in args.providers:
        openai(transforms, args.thresholds)
    if "amazon" in args.providers:
        amazon(transforms, args.thresholds)


if __name__ == "__main__":
    main()
