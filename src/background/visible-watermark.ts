import type { WatermarkScheme } from '../shared/messages';
import geminiLogo from './gemini-logo.json';
import klingLogo from './kling-logo.json';
import type { PixelImage } from './preprocess';
import visibleMarks from './visible-marks.json';

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

const MARK_MIN_LOGO = 20;
const MARK_SCAN_RANGE = [0.04, 0.1];
const MARK_SCAN_STEP = 1.04;
const MARK_THRESHOLDS: Record<keyof typeof visibleMarks.marks, number> = {
  doubao: 0.6,
  jimeng: 0.6,
  qwen: 0.6,
  'kling-cjk': 0.6,
  'grok-2024': 0.8,
  'grok-aurora': 0.65,
  grok: 0.65,
  bing: 0.7,
  meta: 0.6,
};

const DALLE_PALETTE = [
  [255, 255, 102],
  [66, 255, 255],
  [81, 218, 76],
  [255, 110, 60],
  [60, 70, 255],
];
const DALLE_REFERENCE_WIDTH = 1024;
const DALLE_CELL = [21, 20];
const DALLE_INNER = 0.6;
const DALLE_FACTORS = [0.96, 1, 1.04];
const DALLE_SLACK = [0, 1, 2, 3];
const DALLE_MIN_CELL = 3;
const DALLE_TOLERANCE = 64;
const DALLE_THRESHOLD = 0.6;

type Anchor = 'tl' | 'tr' | 'bl' | 'br';
type Placement = [logo: number, horizontal: number, vertical: number];
type Box = (width: number, height: number) => Placement;

interface Layout {
  template: Template;
  anchor: Anchor;
  placements: (width: number, height: number) => Placement[];
  scales: number;
  minLogo: number;
}

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
let klingLayouts: Layout[] | null = null;
let markLayouts: { name: keyof typeof visibleMarks.marks; scheme: WatermarkScheme; layout: Layout }[] | null = null;

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

function loadKling(): Layout[] {
  return Object.entries(klingLogo.layouts).map(([name, { width, height, contrast }]) => {
    const box = KLING_BOXES[name as keyof typeof klingLogo.layouts];
    return {
      template: makeTemplate(Float64Array.from(decode(contrast)), width, height),
      anchor: 'br',
      placements: (w, h) => [box(w, h)],
      scales: KLING_SCALES,
      minLogo: KLING_MIN_LOGO,
    };
  });
}

function scanPlacements([right, bottom]: number[]): (width: number, height: number) => Placement[] {
  return (width, height) => {
    const long = Math.max(width, height);
    const out: Placement[] = [];
    for (let logo = Math.max(MARK_MIN_LOGO, MARK_SCAN_RANGE[0] * long); logo <= MARK_SCAN_RANGE[1] * long; logo *= MARK_SCAN_STEP) {
      out.push([logo, right * logo, bottom * logo]);
    }
    return out;
  };
}

function shortSidePlacements([logo, right, bottom]: number[]): (width: number, height: number) => Placement[] {
  return (width, height) => {
    const short = Math.min(width, height);
    return [[logo * short, right * short, bottom * short]];
  };
}

function loadMarks(): NonNullable<typeof markLayouts> {
  return Object.entries(visibleMarks.marks).map(([name, mark]) => ({
    name: name as keyof typeof visibleMarks.marks,
    scheme: mark.scheme as WatermarkScheme,
    layout: {
      template: makeTemplate(Float64Array.from(decode(mark.contrast)), mark.width, mark.height),
      anchor: 'anchor' in mark ? (mark.anchor as Anchor) : 'br',
      placements: 'short' in mark ? shortSidePlacements(mark.short) : scanPlacements(mark.scan),
      scales: 'short' in mark ? KLING_SCALES : 0,
      minLogo: 'min' in mark ? mark.min : MARK_MIN_LOGO,
    },
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

function gridOrigin(extent: number, margin: number, cells: number, scale: number, shift: number, far: boolean): number | null {
  const start = far ? extent - margin - (cells + PAD - shift) * scale : margin - (PAD - shift) * scale;
  const fits = far ? margin >= (PAD + shift) * scale : start >= 0;
  return start >= 0 && fits ? start : null;
}

function rounded(value: number): number {
  return Math.floor(value + 0.5);
}

function dalleDeviation(image: PixelImage, right: number, bottom: number, cellWidth: number, cellHeight: number): number | null {
  const { width, height, channels, data } = image;
  let worst = 0;
  for (let k = 0; k < 5; k++) {
    const left = right - (5 - k) * cellWidth;
    const x0 = rounded(left + (cellWidth * (1 - DALLE_INNER)) / 2);
    const x1 = Math.max(rounded(left + (cellWidth * (1 + DALLE_INNER)) / 2), x0 + 1);
    const y0 = rounded(bottom - (cellHeight * (1 + DALLE_INNER)) / 2);
    const y1 = Math.max(rounded(bottom - (cellHeight * (1 - DALLE_INNER)) / 2), y0 + 1);
    if (x0 < 0 || y0 < 0 || x1 > width || y1 > height) return null;
    const count = (x1 - x0) * (y1 - y0);
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      let squares = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const v = data[(y * width + x) * channels + c];
          sum += v;
          squares += v * v;
        }
      }
      const mean = sum / count;
      worst = Math.max(worst, Math.abs(mean - DALLE_PALETTE[k][c]), Math.sqrt(Math.max(squares / count - mean * mean, 0)));
    }
  }
  return worst;
}

export function dalleBarScore(image: PixelImage): number {
  let best = -1;
  for (const factor of DALLE_FACTORS) {
    const scale = (image.width / DALLE_REFERENCE_WIDTH) * factor;
    const cellWidth = DALLE_CELL[0] * scale;
    if (cellWidth < DALLE_MIN_CELL) continue;
    for (const sx of DALLE_SLACK) {
      for (const sy of DALLE_SLACK) {
        const worst = dalleDeviation(image, image.width - sx * scale, image.height - sy * scale, cellWidth, DALLE_CELL[1] * scale);
        if (worst !== null) best = Math.max(best, 1 - worst / DALLE_TOLERANCE);
      }
    }
  }
  return best;
}

function layoutScore(image: PixelImage, { template, anchor, placements, scales, minLogo }: Layout): number {
  const { width, height } = image;
  let best = -1;
  for (const [logo, horizontal, vertical] of placements(width, height)) {
    if (logo < minLogo) continue;
    for (let e = -scales; e <= scales; e++) {
      const factor = KLING_STEP ** e;
      const scale = (logo * factor) / template.columns;
      for (const dy of KLING_OFFSETS) {
        for (const dx of KLING_OFFSETS) {
          const x0 = gridOrigin(width, horizontal * factor, template.columns, scale, dx, anchor[1] === 'r');
          const y0 = gridOrigin(height, vertical * factor, template.rows, scale, dy, anchor[0] === 'b');
          if (x0 === null || y0 === null) continue;
          best = Math.max(best, bestCorrelation(featureGrid(image, x0, y0, scale, template), template));
        }
      }
    }
  }
  return best;
}

export function klingLogoScore(image: PixelImage): number {
  klingLayouts ??= loadKling();
  return Math.max(...klingLayouts.map((layout) => layoutScore(image, layout)));
}

export function visibleMarkScores(image: PixelImage): Record<keyof typeof visibleMarks.marks, number> {
  markLayouts ??= loadMarks();
  return Object.fromEntries(markLayouts.map(({ name, layout }) => [name, layoutScore(image, layout)])) as Record<keyof typeof visibleMarks.marks, number>;
}

function visibleMark(image: PixelImage): WatermarkScheme | null {
  markLayouts ??= loadMarks();
  let best: { ratio: number; scheme: WatermarkScheme } | null = null;
  for (const { name, scheme, layout } of markLayouts) {
    const ratio = layoutScore(image, layout) / MARK_THRESHOLDS[name];
    if (ratio >= 1 && (!best || ratio > best.ratio)) best = { ratio, scheme };
  }
  return best?.scheme ?? null;
}

export function detectVisibleWatermark(image: PixelImage): WatermarkScheme | null {
  if (geminiLogoScore(image) >= GEMINI_THRESHOLD) return 'gemini-visible';
  if (klingLogoScore(image) >= KLING_THRESHOLD) return 'kling-visible';
  if (dalleBarScore(image) >= DALLE_THRESHOLD) return 'dalle-visible';
  return visibleMark(image);
}
