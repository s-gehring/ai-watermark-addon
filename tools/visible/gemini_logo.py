import argparse
import base64
import json
import re
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "src" / "background" / "gemini-logo.json"
CACHE = Path(__file__).resolve().parent / ".cache"
SOURCE = "https://raw.githubusercontent.com/allenk/GeminiWatermarkTool/7c6a99f2c3df97eb3c430ef87a0c962aea5cb80e/assets/embedded_assets.hpp"
SIZE = 24
PAD = 2
GRID = SIZE + 2 * PAD
STEP = 1.05
MIN_LOGO = 16
MAX_LOGO = 96
MARGINS = (2 / 3, 2)
HEADROOM = 8
THRESHOLD = 0.88


def masks():
    path = CACHE / "embedded_assets.hpp"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE, path)
    out = {}
    for name, body in re.findall(r"unsigned char (\w+)\[\] = \{([^}]*)\}", path.read_text()):
        png = bytes(int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]{2}", body))
        out[name] = np.asarray(Image.open(BytesIO(png)).convert("RGB"), np.float64).max(axis=2) / 255
    return out


def build():
    alpha = Image.fromarray(masks()["bg_96_png"].astype(np.float32)).resize((SIZE, SIZE), Image.Resampling.BOX)
    quantized = np.round(np.asarray(alpha, np.float64) * 255).astype(np.uint8)
    meta = {"source": SOURCE, "mask": "bg_96_png", "size": SIZE, "alpha": base64.b64encode(quantized.tobytes()).decode()}
    TEMPLATE_PATH.write_text(json.dumps(meta, indent=2) + "\n")


def coordinates():
    c = np.arange(SIZE) - (SIZE - 1) / 2
    return np.tile(c, (SIZE, 1)), np.tile(c[:, None], (1, SIZE))


def template():
    meta = json.loads(TEMPLATE_PATH.read_text())
    alpha = np.frombuffer(base64.b64decode(meta["alpha"]), np.uint8).reshape(SIZE, SIZE) / 255
    t = -np.log1p(-alpha)
    t = t - t.mean()
    x, y = coordinates()
    return t - (t * x).sum() / (x * x).sum() * x - (t * y).sum() / (y * y).sum() * y


def area_weights(start, scale):
    lo = start + np.arange(GRID) * scale
    first, last = int(np.floor(lo[0])), int(np.ceil(lo[-1] + scale))
    i = np.arange(first, last)[None, :]
    w = np.clip(np.minimum(lo[:, None] + scale, i + 1) - np.maximum(lo[:, None], i), 0, None) / scale
    return w, first, last


def feature_grid(plane, x0, y0, scale):
    wx, fx, lx = area_weights(x0, scale)
    wy, fy, ly = area_weights(y0, scale)
    return -np.log(255 + HEADROOM - wy @ plane[fy:ly, fx:lx] @ wx.T)


def best_correlation(grid, t):
    x, y = coordinates()
    windows = np.lib.stride_tricks.sliding_window_view(grid, (SIZE, SIZE))
    s = windows.sum(axis=(2, 3))
    variance = (
        (windows**2).sum(axis=(2, 3))
        - s**2 / SIZE**2
        - np.einsum("ijkl,kl->ij", windows, x) ** 2 / (x * x).sum()
        - np.einsum("ijkl,kl->ij", windows, y) ** 2 / (y * y).sum()
    )
    norm = np.sqrt(np.maximum(variance, 0) * (t * t).sum())
    r = np.where(norm > 1e-9, np.einsum("ijkl,kl->ij", windows, t) / np.maximum(norm, 1e-9), -1)
    return float(r.max())


def score(rgb, t=None):
    t = template() if t is None else t
    height, width, _ = rgb.shape
    plane = rgb.min(axis=2).astype(np.float64)
    best = -1.0
    for margin in MARGINS:
        largest = min(MAX_LOGO, min(width, height) / (1 + margin + PAD / SIZE))
        logo = MIN_LOGO
        while logo <= largest:
            scale = logo / SIZE
            edge = (margin + 1) * logo + PAD * scale
            best = max(best, best_correlation(feature_grid(plane, width - edge, height - edge, scale), t))
            logo *= STEP
    return best


def apply_logo(rgb, alpha, margin):
    out = rgb.astype(np.float64).copy()
    height, width, _ = out.shape
    size = alpha.shape[0]
    y, x = height - margin - size, width - margin - size
    a = alpha[..., None]
    out[y : y + size, x : x + size] = a * 255 + (1 - a) * out[y : y + size, x : x + size]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description="Reference implementation of the visible Gemini logo check.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help=f"write {TEMPLATE_PATH.relative_to(ROOT)} from the masks of GeminiWatermarkTool")
    scorer = sub.add_parser("score", help="print the logo score of each image")
    scorer.add_argument("images", nargs="+")
    args = parser.parse_args()
    if args.command == "build":
        build()
        return
    t = template()
    for path in args.images:
        s = score(load_rgb(path), t)
        print(f"{s:.4f} {'logo' if s >= THRESHOLD else '-   '} {path}")


if __name__ == "__main__":
    main()
