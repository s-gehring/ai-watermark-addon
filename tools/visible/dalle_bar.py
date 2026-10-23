import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = np.array([(255, 255, 102), (66, 255, 255), (81, 218, 76), (255, 110, 60), (60, 70, 255)], np.float64)
REFERENCE_WIDTH = 1024
CELL_WIDTH = 21
CELL_HEIGHT = 20
INNER = 0.6
FACTORS = (0.96, 1.0, 1.04)
SLACK = (0, 1, 2, 3)
MIN_CELL = 3
TOLERANCE = 64
THRESHOLD = 0.6


def rounded(value):
    return int(np.floor(value + 0.5))


def cell_scores(rgb, right, bottom, cell_width, cell_height):
    height, width, _ = rgb.shape
    worst = 0.0
    for k in range(5):
        left = right - (5 - k) * cell_width
        x0, x1 = rounded(left + cell_width * (1 - INNER) / 2), rounded(left + cell_width * (1 + INNER) / 2)
        y0, y1 = rounded(bottom - cell_height * (1 + INNER) / 2), rounded(bottom - cell_height * (1 - INNER) / 2)
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return None
        pixels = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
        worst = max(worst, np.abs(pixels.mean(axis=0) - PALETTE[k]).max(), pixels.std(axis=0).max())
    return worst


def template():
    return None


def score(rgb, t=None):
    height, width, _ = rgb.shape
    best = -1.0
    for factor in FACTORS:
        scale = width / REFERENCE_WIDTH * factor
        cell_width, cell_height = CELL_WIDTH * scale, CELL_HEIGHT * scale
        if cell_width < MIN_CELL:
            continue
        for sx in SLACK:
            for sy in SLACK:
                worst = cell_scores(rgb, width - sx * scale, height - sy * scale, cell_width, cell_height)
                if worst is not None:
                    best = max(best, 1 - worst / TOLERANCE)
    return best


def main():
    parser = argparse.ArgumentParser(description="Reference implementation of the DALL-E 2 color bar check.")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    for path in args.images:
        s = score(np.asarray(Image.open(path).convert("RGB")))
        print(f"{s:.4f} {'bar' if s >= THRESHOLD else '-  '} {Path(path)}")


if __name__ == "__main__":
    main()
