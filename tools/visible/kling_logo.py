import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "src" / "background" / "kling-logo.json"
REAL = Path(__file__).resolve().parent / ".cache" / "real"
PAD = 2
STEP = 1.02
SCALES = 2
OFFSETS = (0, 0.5)
MIN_LOGO = 24
HEADROOM = 8
THRESHOLD = 0.6


def box_2024(width, height):
    long = max(width, height)
    return 0.0898 * long, 0.0205 * long, 0.0166 * long


def box_2026(width, height):
    short, long = min(width, height), max(width, height)
    right = min(0.0142 * long, 0.0252 * short)
    return 0.2005 * short, right, 0.7 * right


LAYOUTS = {
    "2024": {
        "grid": (48, 12),
        "box": box_2024,
        "folder": "kling-2024",
        "source": "https://huggingface.co/datasets/nyuuzyou/klingai/tree/4ae5e439e4e40b6f70c13bcd2f749d78f8416abe",
    },
    "2026": {
        "grid": (64, 18),
        "box": box_2026,
        "folder": "kling-2026",
        "source": "Kling CDN images linked from andreathar/Movies_Projects@35f84474377f86876fcf691dee16036b8601a495 and newmemomedo2018-afk/cgi-generator-mongo@09dcdddac5a93c5f2be7c23020be695dfcd12930",
    },
}


def area_weights(start, scale, cells):
    lo = start + np.arange(cells) * scale
    first, last = int(np.floor(lo[0])), int(np.ceil(lo[-1] + scale))
    i = np.arange(first, last)[None, :]
    w = np.clip(np.minimum(lo[:, None] + scale, i + 1) - np.maximum(lo[:, None], i), 0, None) / scale
    return w, first, last


def placements(layout, width, height):
    columns, rows = LAYOUTS[layout]["grid"]
    logo, right, bottom = LAYOUTS[layout]["box"](width, height)
    if logo < MIN_LOGO:
        return
    for e in range(-SCALES, SCALES + 1):
        factor = STEP**e
        scale = logo * factor / columns
        for dy in OFFSETS:
            for dx in OFFSETS:
                x0 = width - right * factor - (columns + PAD - dx) * scale
                y0 = height - bottom * factor - (rows + PAD - dy) * scale
                if x0 >= 0 and y0 >= 0 and right * factor >= (PAD + dx) * scale and bottom * factor >= (PAD + dy) * scale:
                    yield x0, y0, scale


def feature_grid(plane, x0, y0, scale, columns, rows):
    wx, fx, lx = area_weights(x0, scale, columns + 2 * PAD)
    wy, fy, ly = area_weights(y0, scale, rows + 2 * PAD)
    return -np.log(255 + HEADROOM - wy @ plane[fy:ly, fx:lx] @ wx.T)


def coordinates(columns, rows):
    return np.tile(np.arange(columns) - (columns - 1) / 2, (rows, 1)), np.tile((np.arange(rows) - (rows - 1) / 2)[:, None], (1, columns))


def detrend(t):
    x, y = coordinates(t.shape[1], t.shape[0])
    t = t - t.mean()
    return t - (t * x).sum() / (x * x).sum() * x - (t * y).sum() / (y * y).sum() * y


def template():
    meta = json.loads(TEMPLATE_PATH.read_text())
    out = {}
    for layout, entry in meta["layouts"].items():
        columns, rows = entry["width"], entry["height"]
        contrast = np.frombuffer(base64.b64decode(entry["contrast"]), np.uint8).reshape(rows, columns).astype(np.float64)
        out[layout] = detrend(contrast)
    return out


def best_correlation(grid, t):
    rows, columns = t.shape
    x, y = coordinates(columns, rows)
    n = rows * columns
    windows = np.lib.stride_tricks.sliding_window_view(grid, (rows, columns))
    s = windows.sum(axis=(2, 3))
    variance = (
        (windows**2).sum(axis=(2, 3))
        - s**2 / n
        - np.einsum("ijkl,kl->ij", windows, x) ** 2 / (x * x).sum()
        - np.einsum("ijkl,kl->ij", windows, y) ** 2 / (y * y).sum()
    )
    norm = np.sqrt(np.maximum(variance, 0) * (t * t).sum())
    r = np.where(norm > 1e-9, np.einsum("ijkl,kl->ij", windows, t) / np.maximum(norm, 1e-9), -1)
    return float(r.max())


def layout_scores(rgb, ts=None):
    ts = template() if ts is None else ts
    height, width, _ = rgb.shape
    plane = rgb.min(axis=2).astype(np.float64)
    scores = {}
    for layout, t in ts.items():
        rows, columns = t.shape
        scores[layout] = max(
            (best_correlation(feature_grid(plane, x0, y0, scale, columns, rows), t) for x0, y0, scale in placements(layout, width, height)),
            default=-1.0,
        )
    return scores


def score(rgb, ts=None):
    return max(layout_scores(rgb, ts).values())


def mean_grid(layout, paths):
    columns, rows = LAYOUTS[layout]["grid"]
    total = np.zeros((rows, columns))
    for path in paths:
        rgb = load_rgb(path)
        height, width, _ = rgb.shape
        logo, right, bottom = LAYOUTS[layout]["box"](width, height)
        scale = logo / columns
        grid = feature_grid(rgb.min(axis=2).astype(np.float64), width - right - (columns + PAD) * scale, height - bottom - (rows + PAD) * scale, scale, columns, rows)
        total += grid[PAD:-PAD, PAD:-PAD]
    return total / len(paths)


def contrast(mean):
    background = np.median(np.concatenate([mean[0], mean[-1], mean[:, 0], mean[:, -1]]))
    c = np.clip(mean - background, 0, None)
    c[c < 0.05 * c.max()] = 0
    return np.round(c / c.max() * 255).astype(np.uint8)


def build():
    meta = {"layouts": {}}
    for layout in LAYOUTS:
        paths = sorted((REAL / LAYOUTS[layout]["folder"]).iterdir())
        c = contrast(mean_grid(layout, paths))
        meta["layouts"][layout] = {
            "source": LAYOUTS[layout]["source"],
            "images": len(paths),
            "width": c.shape[1],
            "height": c.shape[0],
            "contrast": base64.b64encode(c.tobytes()).decode(),
        }
    TEMPLATE_PATH.write_text(json.dumps(meta, indent=2) + "\n")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description="Reference implementation of the visible Kling logo check.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help=f"write {TEMPLATE_PATH.relative_to(ROOT)} from the labeled images in {REAL.relative_to(ROOT)}")
    scorer = sub.add_parser("score", help="print the logo score of each image")
    scorer.add_argument("images", nargs="+")
    args = parser.parse_args()
    if args.command == "build":
        build()
        return
    ts = template()
    for path in args.images:
        scores = layout_scores(load_rgb(path), ts)
        s = max(scores.values())
        print(f"{s:.4f} {'logo' if s >= THRESHOLD else '-   '} {' '.join(f'{k} {v:.3f}' for k, v in scores.items())} {path}")


if __name__ == "__main__":
    main()
