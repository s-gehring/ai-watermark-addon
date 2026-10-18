import tile from './bytedance-tile.json';
import type { WatermarkScheme } from '../shared/messages';
import type { PixelImage } from './preprocess';

const TILE_WIDTH = tile.width;
const TILE_HEIGHT = tile.height;
const MIN_TILES = 24;
const CLIP = 8;
const BOX = 5;
const THRESHOLD = 0.5;

const template = (() => {
  const raw = Int8Array.from(atob(tile.tile), (c) => (c.charCodeAt(0) << 24) >> 24);
  return standardized(Float64Array.from(raw));
})();

function standardized(values: Float64Array): Float64Array {
  let mean = 0;
  for (const v of values) mean += v;
  mean /= values.length;
  let variance = 0;
  for (const v of values) variance += (v - mean) ** 2;
  const std = Math.sqrt(variance / values.length) || 1;
  return values.map((v) => (v - mean) / std);
}

function foldedTile(image: PixelImage, columns: number, rows: number): Float64Array {
  const { width, channels, data } = image;
  const sum = new Float64Array(TILE_WIDTH * TILE_HEIGHT);
  const cb = new Float64Array(TILE_WIDTH * TILE_HEIGHT);
  for (let ty = 0; ty < rows; ty++) {
    for (let tx = 0; tx < columns; tx++) {
      let mean = 0;
      for (let y = 0; y < TILE_HEIGHT; y++) {
        let p = ((ty * TILE_HEIGHT + y) * width + tx * TILE_WIDTH) * channels;
        for (let x = 0; x < TILE_WIDTH; x++, p += channels) {
          const v = -0.168736 * data[p] - 0.331264 * data[p + 1] + 0.5 * data[p + 2];
          cb[y * TILE_WIDTH + x] = v;
          mean += v;
        }
      }
      mean /= cb.length;
      for (let i = 0; i < cb.length; i++) {
        const v = cb[i] - mean;
        sum[i] += v < -CLIP ? -CLIP : v > CLIP ? CLIP : v;
      }
    }
  }
  return sum;
}

function circularBox(values: Float64Array, width: number, height: number, horizontal: boolean): Float64Array {
  const out = new Float64Array(values.length);
  const r = (BOX - 1) / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) {
        const xx = horizontal ? (x + k + width) % width : x;
        const yy = horizontal ? y : (y + k + height) % height;
        s += values[yy * width + xx];
      }
      out[y * width + x] = s / BOX;
    }
  }
  return out;
}

export function bytedanceTileScore(image: PixelImage): number | null {
  const columns = Math.floor(image.width / TILE_WIDTH);
  const rows = Math.floor(image.height / TILE_HEIGHT);
  if (columns < 3 || rows < 3 || columns * rows < MIN_TILES) return null;
  const folded = foldedTile(image, columns, rows);
  const blurred = circularBox(circularBox(folded, TILE_WIDTH, TILE_HEIGHT, true), TILE_WIDTH, TILE_HEIGHT, false);
  const residual = standardized(folded.map((v, i) => v - blurred[i]));
  let dot = 0;
  for (let i = 0; i < residual.length; i++) dot += residual[i] * template[i];
  return dot / residual.length;
}

export function detectBytedanceWatermark(image: PixelImage): WatermarkScheme | null {
  return (bytedanceTileScore(image) ?? 0) >= THRESHOLD ? 'bytedance' : null;
}
