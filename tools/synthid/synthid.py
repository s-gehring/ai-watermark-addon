import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 512
MIN_SIDE = 64
MIN_FREQUENCY = 4
NYQUIST_MARGIN = 0.45
GRID_PERIODS = (8, 16)
GRID_TOLERANCE = 1.5
YCBCR = np.array(
    [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
    np.float64,
)


def cubic(x):
    a = -0.5
    x = np.abs(x)
    return np.where(x < 1, ((a + 2) * x - (a + 3)) * x * x + 1, np.where(x < 2, (((x - 5) * x + 8) * x - 4) * a, 0.0))


@lru_cache(maxsize=64)
def resample_matrix(n_in, n_out):
    scale = n_in / n_out
    filter_scale = max(scale, 1.0)
    support = 2.0 * filter_scale
    m = np.zeros((n_out, n_in))
    for o in range(n_out):
        center = (o + 0.5) * scale
        lo = max(int(np.floor(center - support)), 0)
        hi = min(int(np.ceil(center + support)), n_in)
        w = cubic((np.arange(lo, hi) + 0.5 - center) / filter_scale)
        m[o, lo:hi] = w / w.sum()
    return m


def load_rgb(path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float64)


def planes(rgb, size=SIZE):
    height, width, _ = rgb.shape
    ycc = rgb @ YCBCR.T
    ry, rx = resample_matrix(height, size), resample_matrix(width, size)
    out = np.stack([ry @ ycc[..., c] @ rx.T for c in range(3)])
    return out - out.mean(axis=(1, 2), keepdims=True)


def unit_spectrum(rgb, size=SIZE):
    spectrum = np.fft.rfft2(planes(rgb, size))
    return spectrum / (np.abs(spectrum) + 1e-6)


def frequencies(size=SIZE):
    ky = np.fft.fftfreq(size, 1 / size)[:, None]
    kx = np.arange(size // 2 + 1)[None, :]
    return ky, kx


def usable(height, width, size=SIZE):
    ky, kx = frequencies(size)
    rows = np.abs(ky) <= NYQUIST_MARGIN * height
    cols = kx <= NYQUIST_MARGIN * width
    for period in GRID_PERIODS:
        for j in range(1, size):
            fy, fx = j * height / period, j * width / period
            if fy <= size / 2:
                rows &= ~(np.abs(np.abs(ky) - fy) < GRID_TOLERANCE)
            if fx <= size / 2:
                cols &= ~(np.abs(kx - fx) < GRID_TOLERANCE)
    return rows & cols


def build(paths, size=SIZE):
    total, count = None, 0
    for path in paths:
        u = unit_spectrum(load_rgb(path), size)
        total = u if total is None else total + u
        count += 1
    mean = total / count
    ky, kx = frequencies(size)
    weights = np.sqrt(np.maximum(np.abs(mean) ** 2 - 1 / count, 0))
    weights *= ((np.abs(ky) >= MIN_FREQUENCY) & (kx >= MIN_FREQUENCY))[None]
    return weights, np.angle(mean), count


def quantize(weights, phases):
    scale = float(weights.max()) / 255
    w = np.round(weights / scale).astype(np.uint8)
    p = (np.round(phases / (2 * np.pi) * 256).astype(np.int64) % 256).astype(np.uint8)
    return w, p, scale


def dequantize(w, p, scale):
    return w.astype(np.float64) * scale, p.astype(np.float64) * (2 * np.pi / 256)


def score(rgb, weights, phases, size=SIZE):
    height, width, _ = rgb.shape
    if min(height, width) < MIN_SIDE:
        return 0.0
    w = weights * usable(height, width, size)[None]
    u = unit_spectrum(rgb, size)
    signal = (w * np.real(u * np.exp(-1j * phases))).sum()
    energy = (w**2).sum()
    return float(signal / np.sqrt(energy / 2)) if energy > 0 else 0.0


def save(path, w, p):
    Path(path).write_bytes(w.tobytes() + p.tobytes())


def load(bin_path, meta_path):
    meta = json.loads(Path(meta_path).read_text())
    size = meta["size"]
    raw = np.frombuffer(Path(bin_path).read_bytes(), np.uint8)
    shape = (3, size, size // 2 + 1)
    count = int(np.prod(shape))
    return (*dequantize(raw[:count].reshape(shape), raw[count:].reshape(shape), meta["weightScale"]), meta)
