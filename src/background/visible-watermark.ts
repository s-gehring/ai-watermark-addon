import type { WatermarkScheme } from '../shared/messages';
import geminiLogo from './gemini-logo.json';
import klingLogo from './kling-logo.json';
import type { PixelImage } from './preprocess';

const PAD = 2;
const HEADROOM = 8;

const GEMINI_SIZE = 24;
const GEMINI_STEP = 1.05;
const GEMINI_MIN_LOGO = 16;
const GEMINI_MAX_LOGO = 96;
const GEMINI_MARGINS = [2 / 3, 2];
const GEMINI_THRESHOLD = 0.88;

const KLING_STEP = 1.02;
const KLING_SCALES = 2;
const KLING_OFFSETS = [0, 0.5];
const KLING_MIN_LOGO = 24;
const KLING_THRESHOLD = 0.6;

type Box = (width: number, height: number) => [logo: number, right: number, bottom: number];

const KLING_BOXES: Record<keyof typeof klingLogo.layouts, Box> = {
  '2024': (width, height) => {
    const long = Math.max(width, height);
    return [0.0898 * long, 0.0205 * long, 0.0166 * long];
  },
  '2026': (width, height) => {
    const short = Math.min(width, height);
    const right = Math.min(0.0142 * Math.max(width, height), 0.0252 * short);
    return [0.2005 * short, right, 0.7 * right];
  },
};

interface Template {
  columns: number;
  rows: number;
  values: Float64Array;
  energy: number;
  x: Float64Array;
  y: Float64Array;
  xx: number;
  yy: number;
}

interface Taps {
  first: Int32Array;
  weights: Float64Array[];
}

let geminiTemplate: Template | null = null;
let klingTemplates: { box: Box; template: Template }[] | null = null;

function decode(base64: string): Uint8Array {
  return Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
}

function centered(length: number): Float64Array {
  return Float64Array.from({ length }, (_, i) => i - (length - 1) / 2);
}

function makeTemplate(values: Float64Array, columns: number, rows: number): Template {
  const x = centered(columns);
  const y = centered(rows);
  const xx = rows * x.reduce((sum, v) => sum + v * v, 0);
  const yy = columns * y.reduce((sum, v) => sum + v * v, 0);
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  let tx = 0;
  let ty = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      values[i * columns + j] -= mean;
      tx += values[i * columns + j] * x[j];
      ty += values[i * columns + j] * y[i];
    }
  }
  let energy = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      values[i * columns + j] -= (tx * x[j]) / xx + (ty * y[i]) / yy;
      energy += values[i * columns + j] ** 2;
    }
  }
  return { columns, rows, values, energy, x, y, xx, yy };
}

function loadGemini(): Template {
  const values = Float64Array.from(decode(geminiLogo.alpha), (a) => -Math.log1p(-a / 255));
  return makeTemplate(values, GEMINI_SIZE, GEMINI_SIZE);
}

function loadKling(): { box: Box; template: Template }[] {
  return Object.entries(klingLogo.layouts).map(([name, { width, height, contrast }]) => ({
    box: KLING_BOXES[name as keyof typeof klingLogo.layouts],
    template: makeTemplate(Float64Array.from(decode(contrast)), width, height),
  }));
}

function taps(start: number, scale: number, cells: number): Taps {
  const first = new Int32Array(cells);
  const weights: Float64Array[] = [];
  for (let u = 0; u < cells; u++) {
    const lo = start + u * scale;
    const hi = lo + scale;
    first[u] = Math.floor(lo);
    const w = new Float64Array(Math.ceil(hi) - first[u]);
    for (let k = 0; k < w.length; k++) w[k] = (Math.min(hi, first[u] + k + 1) - Math.max(lo, first[u] + k)) / scale;
    weights.push(w);
  }
  return { first, weights };
}

function featureGrid(image: PixelImage, x0: number, y0: number, scale: number, template: Template): Float64Array {
  const { width, channels, data } = image;
  const columnCount = template.columns + 2 * PAD;
  const rowCount = template.rows + 2 * PAD;
  const xs = taps(x0, scale, columnCount);
  const ys = taps(y0, scale, rowCount);
  const top = ys.first[0];
  const rows = ys.first[rowCount - 1] + ys.weights[rowCount - 1].length - top;
  const columns = new Float64Array(rows * columnCount);
  for (let r = 0; r < rows; r++) {
    for (let u = 0; u < columnCount; u++) {
      const w = xs.weights[u];
      let sum = 0;
      for (let k = 0; k < w.length; k++) {
        const p = ((top + r) * width + xs.first[u] + k) * channels;
        sum += w[k] * Math.min(data[p], data[p + 1], data[p + 2]);
      }
      columns[r * columnCount + u] = sum;
    }
  }
  const grid = new Float64Array(rowCount * columnCount);
  for (let v = 0; v < rowCount; v++) {
    const w = ys.weights[v];
    for (let u = 0; u < columnCount; u++) {
      let sum = 0;
      for (let k = 0; k < w.length; k++) sum += w[k] * columns[(ys.first[v] - top + k) * columnCount + u];
      grid[v * columnCount + u] = -Math.log(255 + HEADROOM - sum);
    }
  }
  return grid;
}

function bestCorrelation(grid: Float64Array, { columns, rows, values, energy, x, y, xx, yy }: Template): number {
  const n = columns * rows;
  const stride = columns + 2 * PAD;
  let best = -1;
  for (let dy = 0; dy <= 2 * PAD; dy++) {
    for (let dx = 0; dx <= 2 * PAD; dx++) {
      let sum = 0;
      let squares = 0;
      let dot = 0;
      let sx = 0;
      let sy = 0;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
          const f = grid[(dy + i) * stride + dx + j];
          sum += f;
          squares += f * f;
          dot += f * values[i * columns + j];
          sx += f * x[j];
          sy += f * y[i];
        }
      }
      const variance = squares - (sum * sum) / n - (sx * sx) / xx - (sy * sy) / yy;
      const norm = Math.sqrt(Math.max(variance, 0) * energy);
      if (norm > 1e-9) best = Math.max(best, dot / norm);
    }
  }
  return best;
}

export function geminiLogoScore(image: PixelImage): number {
  geminiTemplate ??= loadGemini();
  let best = -1;
  for (const margin of GEMINI_MARGINS) {
    const largest = Math.min(GEMINI_MAX_LOGO, Math.min(image.width, image.height) / (1 + margin + PAD / GEMINI_SIZE));
    for (let logo = GEMINI_MIN_LOGO; logo <= largest; logo *= GEMINI_STEP) {
      const scale = logo / GEMINI_SIZE;
      const edge = (margin + 1) * logo + PAD * scale;
      best = Math.max(best, bestCorrelation(featureGrid(image, image.width - edge, image.height - edge, scale, geminiTemplate), geminiTemplate));
    }
  }
  return best;
}

export function klingLogoScore(image: PixelImage): number {
  klingTemplates ??= loadKling();
  const { width, height } = image;
  let best = -1;
  for (const { box, template } of klingTemplates) {
    const [logo, right, bottom] = box(width, height);
    if (logo < KLING_MIN_LOGO) continue;
    for (let e = -KLING_SCALES; e <= KLING_SCALES; e++) {
      const factor = KLING_STEP ** e;
      const scale = (logo * factor) / template.columns;
      for (const dy of KLING_OFFSETS) {
        for (const dx of KLING_OFFSETS) {
          const x0 = width - right * factor - (template.columns + PAD - dx) * scale;
          const y0 = height - bottom * factor - (template.rows + PAD - dy) * scale;
          if (x0 < 0 || y0 < 0 || right * factor < (PAD + dx) * scale || bottom * factor < (PAD + dy) * scale) continue;
          best = Math.max(best, bestCorrelation(featureGrid(image, x0, y0, scale, template), template));
        }
      }
    }
  }
  return best;
}

export function detectVisibleWatermark(image: PixelImage): WatermarkScheme | null {
  if (geminiLogoScore(image) >= GEMINI_THRESHOLD) return 'gemini-visible';
  return klingLogoScore(image) >= KLING_THRESHOLD ? 'kling-visible' : null;
}
