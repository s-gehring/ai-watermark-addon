import json
import math
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


def usable_axes(height, width, size=SIZE):
    ky, kx = frequencies(size)
    rows = np.abs(ky[:, 0]) <= NYQUIST_MARGIN * height
    cols = kx[0] <= NYQUIST_MARGIN * width
    for period in GRID_PERIODS:
        for j in range(1, size):
            fy, fx = j * height / period, j * width / period
            if fy <= size / 2:
                rows &= ~(np.abs(np.abs(ky[:, 0]) - fy) < GRID_TOLERANCE)
            if fx <= size / 2:
                cols &= ~(np.abs(kx[0] - fx) < GRID_TOLERANCE)
    return rows, cols


def usable(height, width, size=SIZE):
    rows, cols = usable_axes(height, width, size)
    return rows[:, None] & cols[None, :]


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


CROP_SIZE = 512
ASPECT_TOLERANCE = 0.002
PROFILE_LENGTH = 4 * CROP_SIZE
REFINE_SCORE = 4
REFINE_STEPS = 4
REFINE_DIVISIONS = 8


def crop_hypotheses(width, height, sizes, min_axis, min_both, max_skew):
    """Scale hypotheses (sy, sx, mode) for an image that may be a crop of an output of one of the sizes."""
    seen = []

    def add(sy, sx, mode):
        if not any(abs(sy - a) < 1e-4 and abs(sx - b) < 1e-4 for a, b, _ in seen):
            seen.append((sy, sx, mode))

    ratios = []
    for w, h in sizes:
        if not any(a * h == b * w for a, b in ratios):
            ratios.append((w, h))
    for w, h in sorted(ratios, key=lambda s: s[0] / s[1]):
        r = (width * h) / (height * w)
        if min_axis <= r < 1 - ASPECT_TOLERANCE:
            add(1.0, r, "x")
        elif 1 + ASPECT_TOLERANCE < r <= 1 / min_axis:
            add(1 / r, 1.0, "y")
    for w, h in sizes:
        sy, sx = height / h, width / w
        if min_both <= min(sy, sx) and max(sy, sx) < 1 and abs(sy - sx) <= max_skew:
            add(sy, sx, "xy")
    return seen


def dirichlet(d, n=CROP_SIZE):
    small = np.abs(d) < 1e-9
    magnitude = np.where(small, 1.0, np.sin(np.pi * d) / np.where(small, 1.0, n * np.sin(np.pi * d / n)))
    return np.exp(1j * np.pi * d * (n - 1) / n) * magnitude


def axis_taps(k, scale, ok, wrap, n=CROP_SIZE):
    """Interpolation taps that read frequency k * scale of the crop spectrum, with the phase of the scale."""
    f = k * scale
    base = np.floor(f).astype(np.int64)
    half = n // 2 + 1
    index, weights, norm = [], [], np.zeros(len(k))
    for t in range(1 if scale == 1 else 2):
        m = base + t
        i = m % n if wrap else np.minimum(m, half - 1)
        good = ok[i] if wrap else (m <= half - 1) & ok[i]
        w = np.conj(dirichlet(f - m, n)) * good
        index.append(i)
        weights.append(w)
        norm += np.abs(w) ** 2
    shift = np.exp(-1j * np.pi * (scale - 1) * k / n) / np.sqrt(np.where(norm > 0, norm, 1))
    return index, [w * shift * (norm > 0) for w in weights], norm > 0


def crop_coefficients(u, rows, cols, sy, sx):
    """Template-grid coefficients of a crop that covers (sy, sx) of the original: u is the crop's unit spectrum."""
    ky, kx = frequencies(CROP_SIZE)
    iy, wy, vy = axis_taps(ky[:, 0], sy, rows, True)
    ix, wx, vx = axis_taps(kx[0], sx, cols, False)
    b = np.zeros(u.shape, np.complex128)
    for a in range(len(iy)):
        for c in range(len(ix)):
            b += (wy[a][:, None] * wx[c][None, :])[None] * u[:, iy[a]][:, :, ix[c]]
    return b, vy[:, None] & vx[None, :]


def offset_scores(g, oy, ox):
    ky, kx = frequencies(CROP_SIZE)
    ey = np.exp(-2j * np.pi * np.outer(oy, ky[:, 0]))
    ex = np.exp(-2j * np.pi * np.outer(kx[0], ox))
    return np.real((ey @ g) @ ex)


def best_offset(g, norm, sy, sx, mode):
    """Highest score over all offsets of the crop that the hypothesis allows."""
    n = CROP_SIZE
    if mode != "xy":
        line = np.zeros(PROFILE_LENGTH, np.complex128)
        if mode == "x":
            line[: n // 2 + 1] = g.sum(axis=0)
        else:
            line[frequencies(n)[0][:, 0].astype(np.int64) % PROFILE_LENGTH] = g.sum(axis=1)
        profile = np.real(np.fft.fft(line)) / norm
        j = np.arange(-2, math.ceil((1 - (sx if mode == "x" else sy)) * PROFILE_LENGTH) + 3)
        return float(profile[j % PROFILE_LENGTH].max())
    full = np.zeros((n, n), np.complex128)
    full[:, : n // 2 + 1] = g
    window = np.real(np.fft.fft2(full)) / norm
    ry = np.arange(-1, math.ceil((1 - sy) * n) + 2)
    rx = np.arange(-1, math.ceil((1 - sx) * n) + 2)
    window = window[np.ix_(ry % n, rx % n)]
    i, j = np.unravel_index(int(np.argmax(window)), window.shape)
    if window[i, j] < REFINE_SCORE:
        return float(window[i, j])
    steps = np.arange(-REFINE_STEPS, REFINE_STEPS + 1) / REFINE_DIVISIONS
    return float(offset_scores(g, (ry[i] + steps) / n, (rx[j] + steps) / n).max() / norm)


def crop_scores(u, rows, cols, templates, hypothesis):
    """z per template (weights, phases) for one hypothesis, maximized over the offsets."""
    sy, sx, mode = hypothesis
    channels = [c for c in range(3) if any(w[c].any() for w, _ in templates)]
    b, valid = crop_coefficients(u[channels], rows, cols, sy, sx)
    out = []
    for weights, phases in templates:
        w = weights[channels]
        g = (w * np.exp(-1j * phases[channels]) * b).sum(axis=0)
        energy = float((w**2 * valid[None]).sum())
        out.append(best_offset(g, math.sqrt(energy / 2), sy, sx, mode) if energy > 0 else 0.0)
    return out


def crop_search(rgb, templates, config):
    """Highest z per provider over the crop hypotheses (templates from load_templates, config from synthid.json)."""
    height, width, _ = rgb.shape
    if min(height, width) < MIN_SIDE:
        return {}
    u = None
    rows, cols = usable_axes(height, width, CROP_SIZE)
    scores = {}
    for provider, sizes in config["sizes"].items():
        group = [(t["weights"], t["phases"]) for t in templates if t["meta"].get("cropSearch") and t["meta"]["provider"] == provider]
        if not group:
            continue
        if u is None:
            u = unit_spectrum(rgb, CROP_SIZE)
        hypotheses = crop_hypotheses(width, height, sizes, config["minAxis"], config["minBoth"], config["maxSkew"])
        scores[provider] = max((z for h in hypotheses for z in crop_scores(u, rows, cols, group, h)), default=-np.inf)
    return scores
