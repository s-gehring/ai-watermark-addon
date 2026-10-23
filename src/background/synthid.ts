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
  cropSearch?: boolean;
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

export type SpectrumCache = Map<number, UnitSpectrum>;

function spectrumWith(image: PixelImage, size: number, channels: readonly number[], spectra: SpectrumCache): UnitSpectrum {
  const have = spectra.get(size);
  const missing = channels.filter((c) => !have?.re[c].length);
  if (!missing.length) return have!;
  const fresh = unitSpectrum(image, size, missing);
  if (!have) {
    spectra.set(size, fresh);
    return fresh;
  }
  for (const c of missing) {
    have.re[c] = fresh.re[c];
    have.im[c] = fresh.im[c];
    have.std[c] = fresh.std[c];
  }
  return have;
}

export function detectSynthId(
  image: PixelImage,
  detectors: readonly SynthIdDetector[],
  spectra: SpectrumCache = new Map(),
): TemplateProvider | null {
  if (Math.min(image.width, image.height) < MIN_SIDE) return null;
  const best = new Map<TemplateProvider, number>();

  const matches = (detector: SynthIdDetector, group: readonly SynthIdDetector[]) => {
    const channels = [0, 1, 2].filter((c) => group.some((d) => d.size === detector.size && channelsOf(d).includes(c)));
    const score = templateScore(spectrumWith(image, detector.size, channels, spectra), detector);
    best.set(detector.provider, Math.max(score, best.get(detector.provider) ?? -Infinity));
    return score >= detector.threshold;
  };

  const direct = detectors.filter((d) => !d.prescreen);
  for (const d of direct) if (matches(d, direct)) return d.provider;
  const prescreened = detectors.filter((d) => d.prescreen && (best.get(d.provider) ?? -Infinity) >= PRESCREEN_SCORE);
  for (const d of prescreened) if (matches(d, prescreened)) return d.provider;
  return null;
}

export interface CropSearchConfig {
  threshold: number;
  minAxis: number;
  minBoth: number;
  maxSkew: number;
  sizes: Partial<Record<SynthIdProvider, readonly (readonly number[])[]>>;
}

export type CropMode = 'x' | 'y' | 'xy';

export interface CropHypothesis {
  sy: number;
  sx: number;
  mode: CropMode;
}

const CROP_SIZE = 512;
const ASPECT_TOLERANCE = 0.002;
const PROFILE_LENGTH = 4 * CROP_SIZE;
const REFINE_SCORE = 4;
const REFINE_STEPS = 4;
const REFINE_DIVISIONS = 8;

export function cropHypotheses(
  width: number,
  height: number,
  sizes: readonly (readonly number[])[],
  { minAxis, minBoth, maxSkew }: Pick<CropSearchConfig, 'minAxis' | 'minBoth' | 'maxSkew'>,
): CropHypothesis[] {
  const seen: CropHypothesis[] = [];
  const add = (sy: number, sx: number, mode: CropMode) => {
    if (!seen.some((h) => Math.abs(sy - h.sy) < 1e-4 && Math.abs(sx - h.sx) < 1e-4)) seen.push({ sy, sx, mode });
  };
  const ratios: (readonly number[])[] = [];
  for (const size of sizes) if (!ratios.some(([w, h]) => w * size[1] === h * size[0])) ratios.push(size);
  ratios.sort((a, b) => a[0] * b[1] - b[0] * a[1]);
  for (const [w, h] of ratios) {
    const r = (width * h) / (height * w);
    if (minAxis <= r && r < 1 - ASPECT_TOLERANCE) add(1, r, 'x');
    else if (1 + ASPECT_TOLERANCE < r && r <= 1 / minAxis) add(1 / r, 1, 'y');
  }
  for (const [w, h] of sizes) {
    const sy = height / h;
    const sx = width / w;
    if (minBoth <= Math.min(sy, sx) && Math.max(sy, sx) < 1 && Math.abs(sy - sx) <= maxSkew) add(sy, sx, 'xy');
  }
  return seen;
}

interface SearchTemplate {
  channels: number[];
  re: Float64Array;
  im: Float64Array;
  squares: Float64Array;
}

const searchTemplates = new WeakMap<SynthIdTemplate, SearchTemplate>();

function searchTemplate(template: SynthIdTemplate): SearchTemplate {
  let prepared = searchTemplates.get(template);
  if (!prepared) {
    const channels = channelsOf(template);
    const count = template.size * (template.size / 2 + 1);
    const re = new Float64Array(channels.length * count);
    const im = new Float64Array(channels.length * count);
    const squares = new Float64Array(channels.length * count);
    channels.forEach((c, j) => {
      for (let i = 0; i < count; i++) {
        const w = template.weights[c * count + i] * template.weightScale;
        const p = template.phases[c * count + i];
        re[j * count + i] = w * COS[p];
        im[j * count + i] = -w * SIN[p];
        squares[j * count + i] = w * w;
      }
    });
    prepared = { channels, re, im, squares };
    searchTemplates.set(template, prepared);
  }
  return prepared;
}

interface AxisTaps {
  count: number;
  index: Int32Array[];
  re: Float64Array[];
  im: Float64Array[];
  valid: Uint8Array;
}

function axisTaps(frequency: (i: number) => number, length: number, scale: number, usable: Uint8Array, wrap: boolean): AxisTaps {
  const n = CROP_SIZE;
  const half = n / 2 + 1;
  const count = scale === 1 ? 1 : 2;
  const index = Array.from({ length: count }, () => new Int32Array(length));
  const re = Array.from({ length: count }, () => new Float64Array(length));
  const im = Array.from({ length: count }, () => new Float64Array(length));
  const valid = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    const k = frequency(i);
    const f = k * scale;
    const base = Math.floor(f);
    let norm = 0;
    for (let t = 0; t < count; t++) {
      const m = base + t;
      const j = wrap ? ((m % n) + n) % n : Math.min(m, half - 1);
      index[t][i] = j;
      if (!(wrap ? usable[j] : m <= half - 1 && usable[j])) continue;
      const d = f - m;
      const magnitude = Math.abs(d) < 1e-9 ? 1 : Math.sin(Math.PI * d) / (n * Math.sin((Math.PI * d) / n));
      const angle = (Math.PI * d * (n - 1)) / n;
      re[t][i] = Math.cos(angle) * magnitude;
      im[t][i] = -Math.sin(angle) * magnitude;
      norm += magnitude * magnitude;
    }
    if (norm <= 0) continue;
    valid[i] = 1;
    const angle = (-Math.PI * (scale - 1) * k) / n;
    const sr = Math.cos(angle) / Math.sqrt(norm);
    const si = Math.sin(angle) / Math.sqrt(norm);
    for (let t = 0; t < count; t++) {
      const a = re[t][i];
      const b = im[t][i];
      re[t][i] = a * sr - b * si;
      im[t][i] = a * si + b * sr;
    }
  }
  return { count, index, re, im, valid };
}

const signedFrequency = (i: number) => (i < CROP_SIZE / 2 ? i : i - CROP_SIZE);

function fft2(re: Float64Array, im: Float64Array, n: number): void {
  const lineRe = new Float64Array(n);
  const lineIm = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    lineRe.set(re.subarray(y * n, (y + 1) * n));
    lineIm.set(im.subarray(y * n, (y + 1) * n));
    fft(lineRe, lineIm);
    re.set(lineRe, y * n);
    im.set(lineIm, y * n);
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) {
      lineRe[y] = re[y * n + x];
      lineIm[y] = im[y * n + x];
    }
    fft(lineRe, lineIm);
    for (let y = 0; y < n; y++) {
      re[y * n + x] = lineRe[y];
      im[y * n + x] = lineIm[y];
    }
  }
}

function offsetScores(gRe: Float64Array, gIm: Float64Array, oy: readonly number[], ox: readonly number[]): number[] {
  const n = CROP_SIZE;
  const half = n / 2 + 1;
  const scores: number[] = [];
  const rowRe = new Float64Array(oy.length * half);
  const rowIm = new Float64Array(oy.length * half);
  oy.forEach((o, a) => {
    for (let y = 0; y < n; y++) {
      const angle = -2 * Math.PI * signedFrequency(y) * o;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      for (let x = 0; x < half; x++) {
        const i = y * half + x;
        rowRe[a * half + x] += gRe[i] * c - gIm[i] * s;
        rowIm[a * half + x] += gRe[i] * s + gIm[i] * c;
      }
    }
  });
  for (let a = 0; a < oy.length; a++) {
    for (const o of ox) {
      let sum = 0;
      for (let x = 0; x < half; x++) {
        const angle = -2 * Math.PI * x * o;
        sum += rowRe[a * half + x] * Math.cos(angle) - rowIm[a * half + x] * Math.sin(angle);
      }
      scores.push(sum);
    }
  }
  return scores;
}

function profileScore(lineRe: Float64Array, lineIm: Float64Array, norm: number, free: number): number {
  fft(lineRe, lineIm);
  let best = -Infinity;
  const last = Math.ceil((1 - free) * PROFILE_LENGTH) + 2;
  for (let j = -2; j <= last; j++) best = Math.max(best, lineRe[(j + PROFILE_LENGTH) % PROFILE_LENGTH] / norm);
  return best;
}

function mapScore(gRe: Float64Array, gIm: Float64Array, norm: number, sy: number, sx: number): number {
  const n = CROP_SIZE;
  const half = n / 2 + 1;
  const mapRe = new Float64Array(n * n);
  const mapIm = new Float64Array(n * n);
  for (let y = 0; y < n; y++) {
    mapRe.set(gRe.subarray(y * half, (y + 1) * half), y * n);
    mapIm.set(gIm.subarray(y * half, (y + 1) * half), y * n);
  }
  fft2(mapRe, mapIm, n);
  let best = -Infinity;
  let bestY = 0;
  let bestX = 0;
  const lastY = Math.ceil((1 - sy) * n) + 1;
  const lastX = Math.ceil((1 - sx) * n) + 1;
  for (let oy = -1; oy <= lastY; oy++) {
    for (let ox = -1; ox <= lastX; ox++) {
      const v = mapRe[((oy + n) % n) * n + ((ox + n) % n)] / norm;
      if (v > best) {
        best = v;
        bestY = oy;
        bestX = ox;
      }
    }
  }
  if (best < REFINE_SCORE) return best;
  const fine = (o: number) => Array.from({ length: 2 * REFINE_STEPS + 1 }, (_, i) => (o + (i - REFINE_STEPS) / REFINE_DIVISIONS) / n);
  return Math.max(...offsetScores(gRe, gIm, fine(bestY), fine(bestX))) / norm;
}

interface MaskedTemplate {
  re: Float64Array;
  im: Float64Array;
  rowEnergy: Float64Array;
  columnEnergy: Float64Array;
}

const maskedTemplates = new WeakMap<UnitSpectrum, Map<SearchTemplate, MaskedTemplate>>();

function maskedTemplate(spectrum: UnitSpectrum, template: SearchTemplate): MaskedTemplate {
  let byTemplate = maskedTemplates.get(spectrum);
  if (!byTemplate) maskedTemplates.set(spectrum, (byTemplate = new Map()));
  let masked = byTemplate.get(template);
  if (!masked) {
    const n = CROP_SIZE;
    const half = n / 2 + 1;
    const re = template.re.slice();
    const im = template.im.slice();
    const rowEnergy = new Float64Array(template.channels.length * n);
    const columnEnergy = new Float64Array(half);
    for (let slot = 0; slot < template.channels.length; slot++) {
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < half; x++) {
          const i = (slot * n + y) * half + x;
          const w2 = template.squares[i];
          if (!spectrum.cols[x]) {
            re[i] = 0;
            im[i] = 0;
          } else rowEnergy[slot * n + y] += w2;
          if (spectrum.rows[y]) columnEnergy[x] += w2;
        }
      }
    }
    masked = { re, im, rowEnergy, columnEnergy };
    byTemplate.set(template, masked);
  }
  return masked;
}

function rowProfiles(spectrum: UnitSpectrum, templates: readonly SearchTemplate[], sy: number): number[] {
  const n = CROP_SIZE;
  const half = n / 2 + 1;
  const rows = axisTaps(signedFrequency, n, sy, spectrum.rows, true);
  return templates.map((t) => {
    const { re: tRe, im: tIm, rowEnergy } = maskedTemplate(spectrum, t);
    const lineRe = new Float64Array(PROFILE_LENGTH);
    const lineIm = new Float64Array(PROFILE_LENGTH);
    let energy = 0;
    t.channels.forEach((c, slot) => {
      const uRe = spectrum.re[c];
      const uIm = spectrum.im[c];
      for (let y = 0; y < n; y++) {
        if (!rows.valid[y]) continue;
        const ti = (slot * n + y) * half;
        let sumRe = 0;
        let sumIm = 0;
        for (let a = 0; a < rows.count; a++) {
          const source = rows.index[a][y] * half;
          let dotRe = 0;
          let dotIm = 0;
          for (let x = 0; x < half; x++) {
            const tr = tRe[ti + x];
            const tm = tIm[ti + x];
            const ur = uRe[source + x];
            const um = uIm[source + x];
            dotRe += tr * ur - tm * um;
            dotIm += tr * um + tm * ur;
          }
          const wr = rows.re[a][y];
          const wi = rows.im[a][y];
          sumRe += wr * dotRe - wi * dotIm;
          sumIm += wr * dotIm + wi * dotRe;
        }
        energy += rowEnergy[slot * n + y];
        const at = (signedFrequency(y) + PROFILE_LENGTH) % PROFILE_LENGTH;
        lineRe[at] += sumRe;
        lineIm[at] += sumIm;
      }
    });
    return energy > 0 ? profileScore(lineRe, lineIm, Math.sqrt(energy / 2), sy) : 0;
  });
}

function columnProfiles(spectrum: UnitSpectrum, templates: readonly SearchTemplate[], sx: number): number[] {
  const n = CROP_SIZE;
  const half = n / 2 + 1;
  const cols = axisTaps((i) => i, half, sx, spectrum.cols, false);
  const [i0, i1] = cols.index;
  const [r0, r1] = cols.re;
  const [m0, m1] = cols.im;
  const lines = templates.map((t) => {
    const { columnEnergy } = maskedTemplate(spectrum, t);
    let energy = 0;
    for (let x = 0; x < half; x++) if (cols.valid[x]) energy += columnEnergy[x];
    return { re: new Float64Array(PROFILE_LENGTH), im: new Float64Array(PROFILE_LENGTH), energy };
  });
  const bRe = new Float64Array(half);
  const bIm = new Float64Array(half);
  for (const c of [0, 1, 2]) {
    const slots = templates.map((t) => t.channels.indexOf(c));
    if (slots.every((slot) => slot < 0)) continue;
    const uRe = spectrum.re[c];
    const uIm = spectrum.im[c];
    for (let y = 0; y < n; y++) {
      if (!spectrum.rows[y]) continue;
      const row = y * half;
      for (let x = 0; x < half; x++) {
        const j0 = row + i0[x];
        const j1 = row + i1[x];
        bRe[x] = r0[x] * uRe[j0] - m0[x] * uIm[j0] + r1[x] * uRe[j1] - m1[x] * uIm[j1];
        bIm[x] = r0[x] * uIm[j0] + m0[x] * uRe[j0] + r1[x] * uIm[j1] + m1[x] * uRe[j1];
      }
      for (let k = 0; k < templates.length; k++) {
        if (slots[k] < 0) continue;
        const { re: tRe, im: tIm } = templates[k];
        const lineRe = lines[k].re;
        const lineIm = lines[k].im;
        const ti = (slots[k] * n + y) * half;
        for (let x = 0; x < half; x++) {
          const tr = tRe[ti + x];
          const tm = tIm[ti + x];
          lineRe[x] += tr * bRe[x] - tm * bIm[x];
          lineIm[x] += tr * bIm[x] + tm * bRe[x];
        }
      }
    }
  }
  return lines.map((line) => (line.energy > 0 ? profileScore(line.re, line.im, Math.sqrt(line.energy / 2), sx) : 0));
}

function mapProfiles(spectrum: UnitSpectrum, templates: readonly SearchTemplate[], sy: number, sx: number): number[] {
  const n = CROP_SIZE;
  const half = n / 2 + 1;
  const rows = axisTaps(signedFrequency, n, sy, spectrum.rows, true);
  const cols = axisTaps((i) => i, half, sx, spectrum.cols, false);
  const gRe = templates.map(() => new Float64Array(n * half));
  const gIm = templates.map(() => new Float64Array(n * half));
  const energy = templates.map(() => 0);
  for (const c of [0, 1, 2]) {
    const slots = templates.map((t) => t.channels.indexOf(c) * n * half);
    if (slots.every((slot) => slot < 0)) continue;
    const uRe = spectrum.re[c];
    const uIm = spectrum.im[c];
    for (let y = 0; y < n; y++) {
      if (!rows.valid[y]) continue;
      for (let x = 0; x < half; x++) {
        if (!cols.valid[x]) continue;
        let bRe = 0;
        let bIm = 0;
        for (let a = 0; a < rows.count; a++) {
          const yr = rows.re[a][y];
          const yi = rows.im[a][y];
          const source = rows.index[a][y] * half;
          for (let b = 0; b < cols.count; b++) {
            const wr = yr * cols.re[b][x] - yi * cols.im[b][x];
            const wi = yr * cols.im[b][x] + yi * cols.re[b][x];
            const j = source + cols.index[b][x];
            bRe += wr * uRe[j] - wi * uIm[j];
            bIm += wr * uIm[j] + wi * uRe[j];
          }
        }
        const i = y * half + x;
        for (let k = 0; k < templates.length; k++) {
          if (slots[k] < 0) continue;
          const t = templates[k];
          const ti = slots[k] + i;
          gRe[k][i] += t.re[ti] * bRe - t.im[ti] * bIm;
          gIm[k][i] += t.re[ti] * bIm + t.im[ti] * bRe;
          energy[k] += t.squares[ti];
        }
      }
    }
  }
  return templates.map((_, k) => (energy[k] > 0 ? mapScore(gRe[k], gIm[k], Math.sqrt(energy[k] / 2), sy, sx) : 0));
}

function hypothesisScores(spectrum: UnitSpectrum, templates: readonly SearchTemplate[], { sy, sx, mode }: CropHypothesis): number[] {
  if (mode === 'x') return columnProfiles(spectrum, templates, sx);
  if (mode === 'y') return rowProfiles(spectrum, templates, sy);
  return mapProfiles(spectrum, templates, sy, sx);
}

function searchGroups(image: PixelImage, detectors: readonly SynthIdDetector[], config: CropSearchConfig, spectra: SpectrumCache) {
  return (Object.keys(config.sizes) as SynthIdProvider[]).flatMap((provider) => {
    const templates = detectors.filter((d) => d.cropSearch && d.provider === provider && d.size === CROP_SIZE).map(searchTemplate);
    if (!templates.length) return [];
    const channels = [...new Set(templates.flatMap((t) => t.channels))];
    return [
      {
        provider,
        templates,
        hypotheses: cropHypotheses(image.width, image.height, config.sizes[provider]!, config),
        spectrum: () => spectrumWith(image, CROP_SIZE, channels, spectra),
      },
    ];
  });
}

export function cropSearchScores(
  image: PixelImage,
  detectors: readonly SynthIdDetector[],
  config: CropSearchConfig,
  spectra: SpectrumCache = new Map(),
): Partial<Record<SynthIdProvider, number>> {
  const scores: Partial<Record<SynthIdProvider, number>> = {};
  if (Math.min(image.width, image.height) < MIN_SIDE) return scores;
  for (const { provider, templates, hypotheses, spectrum } of searchGroups(image, detectors, config, spectra)) {
    scores[provider] = Math.max(-Infinity, ...hypotheses.flatMap((h) => hypothesisScores(spectrum(), templates, h)));
  }
  return scores;
}

export function searchCroppedSynthId(
  image: PixelImage,
  detectors: readonly SynthIdDetector[],
  config: CropSearchConfig,
  spectra: SpectrumCache = new Map(),
): SynthIdProvider | null {
  if (Math.min(image.width, image.height) < MIN_SIDE) return null;
  for (const { provider, templates, hypotheses, spectrum } of searchGroups(image, detectors, config, spectra)) {
    for (const h of hypotheses) if (hypothesisScores(spectrum(), templates, h).some((z) => z >= config.threshold)) return provider;
  }
  return null;
}
