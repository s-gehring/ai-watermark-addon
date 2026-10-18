import argparse
import base64
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from kling_logo import PAD, best_correlation, detrend, feature_grid, load_rgb

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "src" / "background" / "visible-marks.json"
CACHE = Path(__file__).resolve().parent / ".cache"
REAL = CACHE / "real"
ALPHA_SOURCE = "https://raw.githubusercontent.com/wiltodelta/remove-ai-watermarks/5677c22a6b1ce7231a7dd0eb324cba6875f37864/src/remove_ai_watermarks/assets/"
COLUMNS = 48
STEP = 1.02
SCALES = 2
OFFSETS = (0, 0.5)
SCAN_STEP = 1.04
MIN_LOGO = 20

MARKS = {
    "doubao": {"scheme": "doubao-visible", "short": (0.1627, 0.0136, 0.0169), "alpha": "doubao_alpha.png", "threshold": 0.6},
    "jimeng": {"scheme": "jimeng-visible", "short": (0.2006, 0.0293, 0.0293), "alpha": "jimeng_alpha.png", "threshold": 0.6},
    "qwen": {"scheme": "qwen-visible", "short": (0.2451, 0.0306, 0.0306), "folder": "qwen", "rows": 10, "threshold": 0.6},
    "kling-cjk": {"scheme": "kling-visible", "short": (0.1615, 0.0303, 0.0236), "folder": "kling-cjk", "rows": 11, "threshold": 0.6},
    "grok-2024": {"scheme": "grok-visible", "scan": (0.062, 0.104), "native": 88.7, "folder": "grok-2024", "rows": 14, "threshold": 0.8},
    "grok-aurora": {"scheme": "grok-visible", "scan": (0.167, 0.167), "native": 65.8, "folder": "grok-aurora", "rows": 18, "threshold": 0.65},
    "grok": {"scheme": "grok-visible", "scan": (0.146, 0.167), "native": 67.8, "folder": "grok", "rows": 17, "threshold": 0.65},
}
SCAN_RANGE = (0.04, 0.10)
THRESHOLD = 1.0


def boxes(mark, width, height):
    if "short" in mark:
        short = min(width, height)
        return [tuple(v * short for v in mark["short"])]
    a, b = mark["scan"]
    lo, hi = SCAN_RANGE
    long = max(width, height)
    out, logo = [], max(MIN_LOGO, lo * long)
    while logo <= hi * long:
        out.append((logo, a * logo, b * logo))
        logo *= SCAN_STEP
    return out


def placements(mark, width, height, columns, rows):
    scales = range(-SCALES, SCALES + 1) if "short" in mark else (0,)
    for logo, right, bottom in boxes(mark, width, height):
        if logo < MIN_LOGO:
            continue
        for e in scales:
            factor = STEP**e
            scale = logo * factor / columns
            for dy in OFFSETS:
                for dx in OFFSETS:
                    x0 = width - right * factor - (columns + PAD - dx) * scale
                    y0 = height - bottom * factor - (rows + PAD - dy) * scale
                    if x0 >= 0 and y0 >= 0 and right * factor >= (PAD + dx) * scale and bottom * factor >= (PAD + dy) * scale:
                        yield x0, y0, scale


def alpha_contrast(name):
    path = CACHE / "alpha" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ALPHA_SOURCE + name, path)
    alpha = Image.open(path).convert("L")
    rows = round(COLUMNS * alpha.height / alpha.width)
    a = np.asarray(Image.fromarray((np.asarray(alpha, np.float64) / 255).astype(np.float32)).resize((COLUMNS, rows), Image.Resampling.BOX), np.float64)
    return -np.log1p(-np.clip(a, 0, 0.95))


def mean_contrast(mark):
    rows = mark["rows"]
    total, paths = np.zeros((rows, COLUMNS)), sorted((REAL / mark["folder"]).iterdir())
    for path in paths:
        rgb = load_rgb(path)
        height, width, _ = rgb.shape
        if "short" in mark:
            logo, right, bottom = boxes(mark, width, height)[0]
        else:
            logo = mark["native"]
            right, bottom = mark["scan"][0] * logo, mark["scan"][1] * logo
        scale = logo / COLUMNS
        x0, y0 = width - right - (COLUMNS + PAD) * scale, height - bottom - (rows + PAD) * scale
        grid = feature_grid(rgb.min(axis=2).astype(np.float64), x0, y0, scale, COLUMNS, rows)[PAD:-PAD, PAD:-PAD]
        total += grid - np.median(np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]]))
    return np.clip(total / len(paths), 0, None), len(paths)


def build():
    meta = {"marks": {}}
    for name, mark in MARKS.items():
        if "alpha" in mark:
            c, source, images = alpha_contrast(mark["alpha"]), ALPHA_SOURCE + mark["alpha"], None
        else:
            c, images = mean_contrast(mark)
            source = mark["folder"]
        q = np.round(c / c.max() * 255).astype(np.uint8)
        layout = {"short": list(mark["short"])} if "short" in mark else {"scan": list(mark["scan"])}
        meta["marks"][name] = {
            "scheme": mark["scheme"],
            "source": source,
            **({"images": images} if images else {}),
            **layout,
            "width": q.shape[1],
            "height": q.shape[0],
            "contrast": base64.b64encode(q.tobytes()).decode(),
        }
    TEMPLATE_PATH.write_text(json.dumps(meta, indent=2) + "\n")


def templates():
    meta = json.loads(TEMPLATE_PATH.read_text())["marks"]
    out = {}
    for name, entry in meta.items():
        c = np.frombuffer(base64.b64decode(entry["contrast"]), np.uint8).reshape(entry["height"], entry["width"]).astype(np.float64)
        out[name] = detrend(c)
    return out


def mark_scores(rgb, ts=None):
    ts = templates() if ts is None else ts
    height, width, _ = rgb.shape
    plane = rgb.min(axis=2).astype(np.float64)
    scores = {}
    for name, t in ts.items():
        rows, columns = t.shape
        scores[name] = max(
            (best_correlation(feature_grid(plane, x0, y0, scale, columns, rows), t) for x0, y0, scale in placements(MARKS[name], width, height, columns, rows)),
            default=-1.0,
        )
    return scores


def template():
    return templates()


def score(rgb, ts=None):
    return max(s / MARKS[name]["threshold"] for name, s in mark_scores(rgb, ts).items())


def detect(rgb, ts=None):
    scores = mark_scores(rgb, ts)
    hits = [(s / MARKS[name]["threshold"], MARKS[name]["scheme"]) for name, s in scores.items() if s >= MARKS[name]["threshold"]]
    return max(hits)[1] if hits else None


def main():
    parser = argparse.ArgumentParser(description="Reference implementation of the visible Doubao, Jimeng, Qwen, Kling (Chinese) and Grok mark checks.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help=f"write {TEMPLATE_PATH.relative_to(ROOT)} from the alpha maps and the labeled images in {REAL.relative_to(ROOT)}")
    scorer = sub.add_parser("score", help="print the mark scores of each image")
    scorer.add_argument("images", nargs="+")
    args = parser.parse_args()
    if args.command == "build":
        build()
        return
    ts = templates()
    for path in args.images:
        rgb = load_rgb(path)
        scores = mark_scores(rgb, ts)
        print(f"{detect(rgb, ts) or '-':15s} {' '.join(f'{k} {v:.3f}' for k, v in scores.items())} {path}")


if __name__ == "__main__":
    main()
