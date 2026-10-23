import argparse
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

import synthid
from build_templates import (
    AMAZON_BORDER,
    AMAZON_MIN_CHANNEL_STD,
    AMAZON_MIN_FREQUENCY,
    AMAZON_SIZE,
    CR,
    OPENAI_BORDER,
    OPENAI_CHANNELS,
    OPENAI_MAX_FREQUENCY,
    OPENAI_TEMPLATES,
    amazon_images,
    openai_images,
)
from eval import PRESCREEN_SCORE, TRANSFORMS

FOLDS = 3
SCORING = {}


def rates(z, thresholds):
    return "  ".join(f"z >= {t:g}: {np.mean(z >= t) * 100:5.1f} %" for t in thresholds)


def accumulate(task):
    size, key, paths = task
    return size, key, synthid.accumulate(paths, size, OPENAI_CHANNELS, masked=True)


def merged(parts):
    return tuple(sum(part[i] for part in parts) for i in range(3))


def openai_templates(parts_by_size):
    templates = []
    for size, parts in parts_by_size.items():
        weights, phases, _ = synthid.finalize(merged(parts), size, OPENAI_CHANNELS, OPENAI_MAX_FREQUENCY, OPENAI_BORDER)
        templates.append((size, *synthid.dequantize(*synthid.quantize(weights, phases))))
    return templates


def initialize(templates):
    SCORING["templates"] = templates


def cascade(templates, rgb):
    scores = [synthid.score(rgb, weights, phases, size) for size, weights, phases in templates]
    return max(scores) if scores[0] >= PRESCREEN_SCORE else scores[0]


def score_openai(task):
    path, names = task
    rgb = synthid.load_rgb(path)
    return {name: cascade(SCORING["templates"], TRANSFORMS[name](rgb)) for name in names}


def report(label, train, templates, paths, transforms, thresholds, workers):
    with Pool(workers, initialize, (templates,)) as pool:
        rows = pool.map(score_openai, [(p, list(transforms)) for p in paths], chunksize=1)
    for name in transforms:
        z = np.array([row[name] for row in rows])
        print(f"{label:10s} train {train:4d}  {name:22s} n={len(z):3d}  median z {np.median(z):5.1f}  {rates(z, thresholds)}", flush=True)


def openai(transforms, thresholds, leave_size_out, max_test, workers):
    groups = {}
    for paths in openai_images().values():
        for path in paths:
            with Image.open(path) as image:
                groups.setdefault(image.size, []).append(path)
    groups = {size: sorted(paths) for size, paths in sorted(groups.items(), key=lambda item: -len(item[1]))}
    folds = {(size, f): paths[f::FOLDS] for size, paths in groups.items() for f in range(FOLDS)}
    with Pool(workers) as pool:
        results = pool.map(accumulate, [(t["size"], key, paths) for t in OPENAI_TEMPLATES for key, paths in folds.items()], chunksize=1)
    parts = {template["size"]: {} for template in OPENAI_TEMPLATES}
    for size, key, result in results:
        parts[size][key] = result

    def build(keep):
        return openai_templates({size: [part for key, part in by_key.items() if keep(key)] for size, by_key in parts.items()})

    held = {size: folds[(size, FOLDS - 1)][:max_test] for size in groups}
    train = sum(len(folds[(size, f)]) for size in groups for f in range(FOLDS - 1))
    templates = build(lambda key: key[1] != FOLDS - 1)
    for size, paths in held.items():
        report(f"{size[0]}x{size[1]}", train, templates, paths, transforms, thresholds, workers)
    report("all sizes", train, templates, [p for paths in held.values() for p in paths], transforms, thresholds, workers)
    if leave_size_out:
        for size, paths in held.items():
            templates = build(lambda key: key[0] != size)
            count = sum(len(v) for key, v in folds.items() if key[0] != size)
            report(f"{size[0]}x{size[1]} out", count, templates, paths, transforms, thresholds, workers)


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
        description="Score held-out images: OpenAI templates from two thirds of the images of every output size, the Amazon template leaving out one repository at a time."
    )
    parser.add_argument("providers", nargs="*", choices=["openai", "amazon"], default=["openai", "amazon"])
    parser.add_argument("--transforms", action="store_true", help="also score re-encoded, rescaled and cropped copies")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[8, 9, 10, 12])
    parser.add_argument("--leave-size-out", action="store_true", help="OpenAI: also build each template without the tested output size")
    parser.add_argument("--max-test", type=int, default=60, help="OpenAI: test images per output size")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    transforms = TRANSFORMS if args.transforms else {"original": TRANSFORMS["original"]}
    if "openai" in args.providers:
        openai(transforms, args.thresholds, args.leave_size_out, args.max_test, args.workers)
    if "amazon" in args.providers:
        amazon(transforms, args.thresholds)


if __name__ == "__main__":
    main()
