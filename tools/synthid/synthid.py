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
LOCAL_CONTRAST_OFFSET = 0.5
TEMPLATE_DIR = Path("src") / "background" / "synthid-templates"
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


def planes(rgb, size=SIZE, channels=(0, 1, 2)):
    height, width, _ = rgb.shape
    ycc = rgb @ YCBCR.T
    ry, rx = resample_matrix(height, size), resample_matrix(width, size)
    out = np.stack([ry @ ycc[..., c] @ rx.T for c in channels])
    return out - out.mean(axis=(1, 2), keepdims=True)


def local_contrast(p):
    n = p.shape[1]
    padded = np.pad(p, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    windows = [padded[:, dy : dy + n, dx : dx + n] for dy in range(3) for dx in range(3)]
    mean = sum(windows) / 9
    variance = sum(w * w for w in windows) / 9 - mean * mean
    q = (p - mean) / (np.sqrt(np.maximum(variance, 0)) + LOCAL_CONTRAST_OFFSET)
    return q - q.mean(axis=(1, 2), keepdims=True)


def spectrum(rgb, size=SIZE, channels=(0, 1, 2)):
    p = planes(rgb, size, channels)
    f = np.fft.rfft2(local_contrast(p))
    u = np.zeros((3, *f.shape[1:]), np.complex128)
    std = np.zeros(3)
    u[list(channels)] = f / (np.abs(f) + 1e-6)
    std[list(channels)] = p.std(axis=(1, 2))
    return u, std


def unit_spectrum(rgb, size=SIZE, channels=(0, 1, 2)):
    return spectrum(rgb, size, channels)[0]


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


def accumulate(paths, size=SIZE, channels=(0, 1, 2), masked=False):
    total, support, count = 0, 0, 0
    for path in paths:
        rgb = load_rgb(path)
        u = unit_spectrum(rgb, size, channels)[list(channels)]
        valid = usable(*rgb.shape[:2], size) if masked else np.ones(u.shape[1:], bool)
        total = total + u * valid
        support = support + valid
        count += 1
    return total, support, count


def finalize(sums, size=SIZE, channels=(0, 1, 2), max_frequency=None, border=0, min_frequency=None):
    total, support, count = sums
    mean = np.divide(total, support, out=np.zeros_like(total), where=support > 0)
    weights = np.zeros((3, size, size // 2 + 1))
    phases = np.zeros_like(weights)
    weights[list(channels)] = np.sqrt(np.maximum(np.abs(mean) ** 2 - 1 / np.maximum(support, 1), 0))
    phases[list(channels)] = np.angle(mean)
    ky, kx = frequencies(size)
    keep = (np.abs(ky) >= MIN_FREQUENCY) & (kx >= MIN_FREQUENCY)
    if max_frequency is not None:
        keep &= np.hypot(ky, kx) <= max_frequency
    if min_frequency is not None:
        keep &= np.hypot(ky, kx) >= min_frequency
    if border:
        weights, phases = without_border(weights, phases, border)
    return weights * keep[None], phases, count


def build(paths, size=SIZE, max_frequency=None, border=0, min_frequency=None, channels=(0, 1, 2), masked=False):
    sums = accumulate(paths, size, channels, masked)
    return finalize(sums, size, channels, max_frequency, border, min_frequency)


def without_border(weights, phases, border):
    size = weights.shape[1]
    spatial = np.fft.irfft2(weights * np.exp(1j * phases), s=(size, size))
    m = round(border * size)
    spatial[:, :m] = spatial[:, -m:] = spatial[:, :, :m] = spatial[:, :, -m:] = 0
    masked = np.fft.rfft2(spatial)
    return np.abs(masked), np.angle(masked)


def quantize(weights, phases):
    scale = float(weights.max()) / 255
    w = np.round(weights / scale).astype(np.uint8)
    p = np.where(w > 0, np.round(phases / (2 * np.pi) * 256).astype(np.int64) % 256, 0).astype(np.uint8)
    return w, p, scale


def dequantize(w, p, scale):
    return w.astype(np.float64) * scale, p.astype(np.float64) * (2 * np.pi / 256)


def score_spectrum(u, height, width, weights, phases, channel_std=None, min_channel_std=0):
    size = u.shape[1]
    w = weights * usable(height, width, size)[None]
    if min_channel_std:
        w = w * (channel_std >= min_channel_std)[:, None, None]
    signal = (w * np.real(u * np.exp(-1j * phases))).sum()
    energy = (w**2).sum()
    return float(signal / np.sqrt(energy / 2)) if energy > 0 else 0.0


def score(rgb, weights, phases, size=SIZE, min_channel_std=0):
    height, width, _ = rgb.shape
    if min(height, width) < MIN_SIDE:
        return 0.0
    u, channel_std = spectrum(rgb, size, tuple(c for c in range(3) if weights[c].any()) or (0, 1, 2))
    return score_spectrum(u, height, width, weights, phases, channel_std, min_channel_std)


def save(path, w, p):
    Path(path).write_bytes(w.tobytes() + p.tobytes())


def load_templates(root):
    templates = []
    for meta in json.loads((Path(root) / "synthid.json").read_text())["templates"]:
        size = meta["size"]
        raw = np.frombuffer((Path(root) / TEMPLATE_DIR / meta["file"]).read_bytes(), np.uint8)
        shape = (3, size, size // 2 + 1)
        count = int(np.prod(shape))
        weights, phases = dequantize(raw[:count].reshape(shape), raw[count:].reshape(shape), meta["weightScale"])
        templates.append({"meta": meta, "weights": weights, "phases": phases})
    return templates
