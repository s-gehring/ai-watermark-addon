import type { WatermarkScheme } from '../shared/messages';
import geminiLogo from './gemini-logo.json';
import type { PixelImage } from './preprocess';

const SIZE = 24;
const PAD = 2;
const GRID = SIZE + 2 * PAD;
const STEP = 1.05;
const MIN_LOGO = 16;
const MAX_LOGO = 96;
const MARGINS = [2 / 3, 2];
const HEADROOM = 8;
const THRESHOLD = 0.88;

interface Template {
  values: Float64Array;
  energy: number;
  x: Float64Array;
  xx: number;
}

interface Taps {
  first: Int32Array;
  weights: Float64Array[];
}

let template: Template | null = null;

function loadTemplate(): Template {
  const alpha = Uint8Array.from(atob(geminiLogo.alpha), (c) => c.charCodeAt(0));
  const values = Float64Array.from(alpha, (a) => -Math.log1p(-a / 255));
  const x = Float64Array.from({ length: SIZE }, (_, i) => i - (SIZE - 1) / 2);
  const xx = SIZE * x.reduce((sum, v) => sum + v * v, 0);
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  let tx = 0;
  let ty = 0;
  for (let i = 0; i < SIZE; i++) {
    for (let j = 0; j < SIZE; j++) {
      values[i * SIZE + j] -= mean;
      tx += values[i * SIZE + j] * x[j];
      ty += values[i * SIZE + j] * x[i];
    }
  }
  let energy = 0;
  for (let i = 0; i < SIZE; i++) {
    for (let j = 0; j < SIZE; j++) {
      values[i * SIZE + j] -= (tx * x[j] + ty * x[i]) / xx;
      energy += values[i * SIZE + j] ** 2;
    }
  }
  return { values, energy, x, xx };
}

function taps(start: number, scale: number): Taps {
  const first = new Int32Array(GRID);
  const weights: Float64Array[] = [];
  for (let u = 0; u < GRID; u++) {
    const lo = start + u * scale;
    const hi = lo + scale;
    first[u] = Math.floor(lo);
    const w = new Float64Array(Math.ceil(hi) - first[u]);
    for (let k = 0; k < w.length; k++) w[k] = (Math.min(hi, first[u] + k + 1) - Math.max(lo, first[u] + k)) / scale;
    weights.push(w);
  }
  return { first, weights };
}

function featureGrid(image: PixelImage, x0: number, y0: number, scale: number): Float64Array {
  const { width, channels, data } = image;
  const xs = taps(x0, scale);
  const ys = taps(y0, scale);
  const top = ys.first[0];
  const rows = ys.first[GRID - 1] + ys.weights[GRID - 1].length - top;
  const columns = new Float64Array(rows * GRID);
  for (let r = 0; r < rows; r++) {
    for (let u = 0; u < GRID; u++) {
      const w = xs.weights[u];
      let sum = 0;
      for (let k = 0; k < w.length; k++) {
        const p = ((top + r) * width + xs.first[u] + k) * channels;
        sum += w[k] * Math.min(data[p], data[p + 1], data[p + 2]);
      }
      columns[r * GRID + u] = sum;
    }
  }
  const grid = new Float64Array(GRID * GRID);
  for (let v = 0; v < GRID; v++) {
    const w = ys.weights[v];
    for (let u = 0; u < GRID; u++) {
      let sum = 0;
      for (let k = 0; k < w.length; k++) sum += w[k] * columns[(ys.first[v] - top + k) * GRID + u];
      grid[v * GRID + u] = -Math.log(255 + HEADROOM - sum);
    }
  }
  return grid;
}

function bestCorrelation(grid: Float64Array, { values, energy, x, xx }: Template): number {
  const n = SIZE * SIZE;
  let best = -1;
  for (let dy = 0; dy <= 2 * PAD; dy++) {
    for (let dx = 0; dx <= 2 * PAD; dx++) {
      let sum = 0;
      let squares = 0;
      let dot = 0;
      let sx = 0;
      let sy = 0;
      for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
          const f = grid[(dy + i) * GRID + dx + j];
          sum += f;
          squares += f * f;
          dot += f * values[i * SIZE + j];
          sx += f * x[j];
          sy += f * x[i];
        }
      }
      const variance = squares - (sum * sum) / n - (sx * sx + sy * sy) / xx;
      const norm = Math.sqrt(Math.max(variance, 0) * energy);
      if (norm > 1e-9) best = Math.max(best, dot / norm);
    }
  }
  return best;
}

export function geminiLogoScore(image: PixelImage): number {
  template ??= loadTemplate();
  let best = -1;
  for (const margin of MARGINS) {
    const largest = Math.min(MAX_LOGO, Math.min(image.width, image.height) / (1 + margin + PAD / SIZE));
    for (let logo = MIN_LOGO; logo <= largest; logo *= STEP) {
      const scale = logo / SIZE;
      const edge = (margin + 1) * logo + PAD * scale;
      best = Math.max(best, bestCorrelation(featureGrid(image, image.width - edge, image.height - edge, scale), template));
    }
  }
  return best;
}

export function detectVisibleWatermark(image: PixelImage): WatermarkScheme | null {
  return geminiLogoScore(image) >= THRESHOLD ? 'gemini-visible' : null;
}
