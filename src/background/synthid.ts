import type { PixelImage } from './preprocess';

export interface SynthIdTemplate {
  size: number;
  weightScale: number;
  weights: Uint8Array;
  phases: Uint8Array;
}

const MIN_SIDE = 64;
const NYQUIST_MARGIN = 0.45;
const GRID_PERIODS = [8, 16];
const GRID_TOLERANCE = 1.5;
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

export function resampledPlanes(image: PixelImage, size: number): Float64Array[] {
  const { width, height, channels, data } = image;
  const horizontal = kernel(width, size);
  const vertical = kernel(height, size);
  const rows = YCBCR.map(() => new Float64Array(height * size));
  const line = YCBCR.map(() => new Float64Array(width));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * channels;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      for (let c = 0; c < 3; c++) line[c][x] = YCBCR[c][0] * r + YCBCR[c][1] * g + YCBCR[c][2] * b;
    }
    for (let o = 0; o < size; o++) {
      const lo = horizontal.start[o];
      const offset = o * horizontal.taps;
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let i = 0; i < horizontal.length[o]; i++) sum += horizontal.weights[offset + i] * line[c][lo + i];
        rows[c][y * size + o] = sum;
      }
    }
  }
  return rows.map((row) => {
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

export function synthIdScore(image: PixelImage, template: SynthIdTemplate): number {
  if (Math.min(image.width, image.height) < MIN_SIDE) return 0;
  const { size, weightScale, weights, phases } = template;
  const half = size / 2 + 1;
  const rows = usableFrequencies(image.height, size, size, true);
  const cols = usableFrequencies(image.width, half, size, false);
  let signal = 0;
  let energy = 0;
  resampledPlanes(image, size).forEach((plane, c) => {
    const { re, im } = rfft2(plane, size);
    for (let y = 0; y < size; y++) {
      if (!rows[y]) continue;
      for (let x = 0; x < half; x++) {
        const t = (c * size + y) * half + x;
        if (!cols[x] || !weights[t]) continue;
        const i = y * half + x;
        const w = weights[t] * weightScale;
        const phase = (phases[t] * 2 * Math.PI) / 256;
        signal += (w * (re[i] * Math.cos(phase) + im[i] * Math.sin(phase))) / (Math.hypot(re[i], im[i]) + 1e-6);
        energy += w * w;
      }
    }
  });
  return energy > 0 ? signal / Math.sqrt(energy / 2) : 0;
}
