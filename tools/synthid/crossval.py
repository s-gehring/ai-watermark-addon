import argparse

import numpy as np

import synthid
from build_templates import OPENAI_BORDER, OPENAI_MAX_FREQUENCY, OPENAI_SIZES, openai_images
from eval import TRANSFORMS


def main():
    parser = argparse.ArgumentParser(description="Build each OpenAI template from two thirds of its images and score the held-out third.")
    parser.add_argument("--transforms", action="store_true", help="also score re-encoded, rescaled and cropped copies")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[8, 9, 10, 12])
    args = parser.parse_args()

    transforms = TRANSFORMS if args.transforms else {"original": TRANSFORMS["original"]}
    for (width, height), paths_by_dataset in openai_images().items():
        paths = sorted(p for group in paths_by_dataset.values() for p in group)
        train, test = [p for i, p in enumerate(paths) if i % 3 != 2], [p for i, p in enumerate(paths) if i % 3 == 2]
        size = OPENAI_SIZES[(width, height)]
        weights, phases, _ = synthid.build(train, size, OPENAI_MAX_FREQUENCY, OPENAI_BORDER)
        weights, phases = synthid.dequantize(*synthid.quantize(weights, phases))
        for name, transform in transforms.items():
            z = np.array([synthid.score(transform(synthid.load_rgb(p)), weights, phases, size) for p in test])
            rates = "  ".join(f"z >= {t:g}: {np.mean(z >= t) * 100:5.1f} %" for t in args.thresholds)
            print(f"{width}x{height}  train {len(train):3d}  {name:16s} n={len(z):3d}  median z {np.median(z):5.1f}  {rates}", flush=True)


if __name__ == "__main__":
    main()
