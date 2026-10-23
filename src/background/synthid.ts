import type { SynthIdProvider } from '../shared/messages';
import type { PixelImage } from './preprocess';

export interface SynthIdTemplate {
  size: number;
  weightScale: number;
  weights: Uint8Array;
  phases: Uint8Array;
  minChannelStd?: number;
}

export type TemplateProvider = SynthIdProvider | 'amazon';

export interface SynthIdDetector extends SynthIdTemplate {
  provider: TemplateProvider;
  threshold: number;
  prescreen?: boolean;
}

const MIN_SIDE = 64;
const NYQUIST_MARGIN = 0.45;
const GRID_PERIODS = [8, 16];
const GRID_TOLERANCE = 1.5;
const PRESCREEN_SCORE = 3;
const LOCAL_CONTRAST_OFFSET = 0.5;
const YCBCR = [
  [0.299, 0.587, 0.114],
  [-0.168736, -0.331264, 0.5],
  [0.5, -0.418688, -0.081312],
] as const;

function cubic(x: number): number {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

interface Kernel {
  start: Int32Array;
  length: Int32Array;
  taps: number;
  weights: Float64Array;
}

function kernel(inSize: number, outSize: number): Kernel {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = 2 * filterScale;
  const taps = Math.ceil(2 * support) + 2;
  const start = new Int32Array(outSize);
  const length = new Int32Array(outSize);
  const weights = new Float64Array(outSize * taps);
  for (let o = 0; o < outSize; o++) {
    const center = (o + 0.5) * scale;
    const lo = Math.max(Math.floor(center - support), 0);
    const hi = Math.min(Math.ceil(center + support), inSize);
    let sum = 0;
    for (let i = lo; i < hi; i++) {
      const w = cubic((i + 0.5 - center) / filterScale);
      weights[o * taps + i - lo] = w;
      sum += w;
    }
    for (let i = 0; i < hi - lo; i++) weights[o * taps + i] /= sum;
    start[o] = lo;
    length[o] = hi - lo;
  }
  return { start, length, taps, weights };
}

export function resampledPlanes(image: PixelImage, size: number, planes: readonly number[] = [0, 1, 2]): Float64Array[] {
  const { width, height, channels, data } = image;
  const horizontal = kernel(width, size);
  const vertical = kernel(height, size);
  const rows = YCBCR.map((_, c) => new Float64Array(planes.includes(c) ? height * size : 0));
  const line = YCBCR.map(() => new Float64Array(width));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * channels;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      for (const c of planes) line[c][x] = YCBCR[c][0] * r + YCBCR[c][1] * g + YCBCR[c][2] * b;
    }
    for (let o = 0; o < size; o++) {
      const lo = horizontal.start[o];
      const offset = o * horizontal.taps;
      for (const c of planes) {
        let sum = 0;
        for (let i = 0; i < horizontal.length[o]; i++) sum += horizontal.weights[offset + i] * line[c][lo + i];
        rows[c][y * size + o] = sum;
      }
    }
  }
  return rows.map((row) => {
    if (row.length === 0) return row;
    const plane = new Float64Array(size * size);
    for (let o = 0; o < size; o++) {
      const lo = vertical.start[o];
      const offset = o * vertical.taps;
      for (let i = 0; i < vertical.length[o]; i++) {
        const w = vertical.weights[offset + i];
        const source = (lo + i) * size;
        for (let x = 0; x < size; x++) plane[o * size + x] += w * row[source + x];
      }
    }
    let mean = 0;
    for (const v of plane) mean += v;
    mean /= plane.length;
    for (let i = 0; i < plane.length; i++) plane[i] -= mean;
    return plane;
  });
}

export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    for (let k = 0; k < len / 2; k++) {
      const wr = Math.cos(angle * k);
      const wi = Math.sin(angle * k);
      for (let i = k; i < n; i += len) {
        const j = i + len / 2;
        const tr = re[j] * wr - im[j] * wi;
        const ti = re[j] * wi + im[j] * wr;
        re[j] = re[i] - tr;
        im[j] = im[i] - ti;
        re[i] += tr;
        im[i] += ti;
      }
    }
  }
}

export function rfft2(plane: Float64Array, size: number): { re: Float64Array; im: Float64Array } {
  const half = size / 2 + 1;
  const re = new Float64Array(size * half);
  const im = new Float64Array(size * half);
  const bufRe = new Float64Array(size);
  const bufIm = new Float64Array(size);
  for (let y = 0; y < size; y++) {
    bufRe.set(plane.subarray(y * size, (y + 1) * size));
    bufIm.fill(0);
    fft(bufRe, bufIm);
    re.set(bufRe.subarray(0, half), y * half);
    im.set(bufIm.subarray(0, half), y * half);
  }
  for (let x = 0; x < half; x++) {
    for (let y = 0; y < size; y++) {
      bufRe[y] = re[y * half + x];
      bufIm[y] = im[y * half + x];
    }
    fft(bufRe, bufIm);
    for (let y = 0; y < size; y++) {
      re[y * half + x] = bufRe[y];
      im[y * half + x] = bufIm[y];
    }
  }
  return { re, im };
}

function reflect(i: number, n: number): number {
  return i < 0 ? -i : i >= n ? 2 * n - 2 - i : i;
}

export function localContrast(plane: Float64Array, size: number): Float64Array {
  const out = new Float64Array(size * size);
  let total = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let sum = 0;
      let squares = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const row = reflect(y + dy, size) * size;
        for (let dx = -1; dx <= 1; dx++) {
          const v = plane[row + reflect(x + dx, size)];
          sum += v;
          squares += v * v;
        }
      }
      const mean = sum / 9;
      const v = (plane[y * size + x] - mean) / (Math.sqrt(Math.max(squares / 9 - mean * mean, 0)) + LOCAL_CONTRAST_OFFSET);
      out[y * size + x] = v;
      total += v;
    }
  }
  const mean = total / out.length;
  for (let i = 0; i < out.length; i++) out[i] -= mean;
  return out;
}

function usableFrequencies(imageSize: number, count: number, size: number, signed: boolean): Uint8Array {
  const usable = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const k = signed && i >= size / 2 ? size - i : i;
    usable[i] = k <= NYQUIST_MARGIN * imageSize ? 1 : 0;
    for (const period of GRID_PERIODS) {
      for (let j = 1; (j * imageSize) / period <= size / 2; j++) {
        if (Math.abs(k - (j * imageSize) / period) < GRID_TOLERANCE) usable[i] = 0;
      }
    }
  }
  return usable;
}

interface UnitSpectrum {
  size: number;
  rows: Uint8Array;
  cols: Uint8Array;
  std: number[];
  re: Float64Array[];
  im: Float64Array[];
}

function unitSpectrum(image: PixelImage, size: number, wanted: readonly number[]): UnitSpectrum {
  const planes = resampledPlanes(image, size, wanted);
  const std = planes.map((plane) => (plane.length ? Math.sqrt(plane.reduce((sum, v) => sum + v * v, 0) / plane.length) : 0));
  const channels = planes.map((plane) => {
    if (!plane.length) return { re: new Float64Array(0), im: new Float64Array(0) };
    const { re, im } = rfft2(localContrast(plane, size), size);
    for (let i = 0; i < re.length; i++) {
      const norm = Math.hypot(re[i], im[i]) + 1e-6;
      re[i] /= norm;
      im[i] /= norm;
    }
    return { re, im };
  });
  return {
    size,
    rows: usableFrequencies(image.height, size, size, true),
    cols: usableFrequencies(image.width, size / 2 + 1, size, false),
    std,
    re: channels.map((c) => c.re),
    im: channels.map((c) => c.im),
  };
}

const usedChannels = new WeakMap<SynthIdTemplate, number[]>();

function channelsOf(template: SynthIdTemplate): number[] {
  let channels = usedChannels.get(template);
  if (!channels) {
    const count = template.weights.length / 3;
    channels = [0, 1, 2].filter((c) => template.weights.subarray(c * count, (c + 1) * count).some((w) => w > 0));
    usedChannels.set(template, channels);
  }
  return channels;
}

const COS = Float64Array.from({ length: 256 }, (_, p) => Math.cos((p * 2 * Math.PI) / 256));
const SIN = Float64Array.from({ length: 256 }, (_, p) => Math.sin((p * 2 * Math.PI) / 256));

function templateScore(spectrum: UnitSpectrum, template: SynthIdTemplate): number {
  const { size, rows, cols } = spectrum;
  const { weightScale, weights, phases } = template;
  const half = size / 2 + 1;
  let signal = 0;
  let energy = 0;
  for (let c = 0; c < 3; c++) {
    if (!spectrum.re[c].length || spectrum.std[c] < (template.minChannelStd ?? 0)) continue;
    const re = spectrum.re[c];
    const im = spectrum.im[c];
    for (let y = 0; y < size; y++) {
      if (!rows[y]) continue;
      for (let x = 0; x < half; x++) {
        const t = (c * size + y) * half + x;
        if (!cols[x] || !weights[t]) continue;
        const i = y * half + x;
        const w = weights[t] * weightScale;
        signal += w * (re[i] * COS[phases[t]] + im[i] * SIN[phases[t]]);
        energy += w * w;
      }
    }
  }
  return energy > 0 ? signal / Math.sqrt(energy / 2) : 0;
}

export function synthIdScore(image: PixelImage, template: SynthIdTemplate): number {
  if (Math.min(image.width, image.height) < MIN_SIDE) return 0;
  return templateScore(unitSpectrum(image, template.size, channelsOf(template)), template);
}

export function detectSynthId(image: PixelImage, detectors: readonly SynthIdDetector[]): TemplateProvider | null {
  if (Math.min(image.width, image.height) < MIN_SIDE) return null;
  const spectra = new Map<number, UnitSpectrum>();
  const best = new Map<TemplateProvider, number>();

  const matches = (detector: SynthIdDetector, group: readonly SynthIdDetector[]) => {
    const have = spectra.get(detector.size);
    const missing = [0, 1, 2].filter(
      (c) => !have?.re[c].length && group.some((d) => d.size === detector.size && channelsOf(d).includes(c)),
    );
    if (missing.length) {
      const fresh = unitSpectrum(image, detector.size, missing);
      if (have) {
        for (const c of missing) {
          have.re[c] = fresh.re[c];
          have.im[c] = fresh.im[c];
          have.std[c] = fresh.std[c];
        }
      } else spectra.set(detector.size, fresh);
    }
    const score = templateScore(spectra.get(detector.size)!, detector);
    best.set(detector.provider, Math.max(score, best.get(detector.provider) ?? -Infinity));
    return score >= detector.threshold;
  };

  const direct = detectors.filter((d) => !d.prescreen);
  for (const d of direct) if (matches(d, direct)) return d.provider;
  const prescreened = detectors.filter((d) => d.prescreen && (best.get(d.provider) ?? -Infinity) >= PRESCREEN_SCORE);
  for (const d of prescreened) if (matches(d, prescreened)) return d.provider;
  return null;
}
