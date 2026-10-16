import type { WatermarkScheme } from '../shared/messages';
import type { PixelImage } from './preprocess';

interface Payload {
  scheme: WatermarkScheme;
  bits: Uint8Array;
}

const MIN_PIXELS = 256 * 256;
const BLOCK = 4;
const SCALE = 36;
const MIN_Z = 5;

function bitsOfBytes(text: string): Uint8Array {
  const bytes = new TextEncoder().encode(text);
  const bits = new Uint8Array(bytes.length * 8);
  bytes.forEach((byte, i) => {
    for (let b = 0; b < 8; b++) bits[i * 8 + b] = (byte >> (7 - b)) & 1;
  });
  return bits;
}

function bitsOfNumber(value: bigint): Uint8Array {
  return Uint8Array.from(Array.from(value.toString(2), Number));
}

const PAYLOADS: Payload[] = [
  { scheme: 'stable-diffusion', bits: bitsOfBytes('StableDiffusionV1') },
  { scheme: 'stable-diffusion', bits: bitsOfBytes('SDV2') },
  { scheme: 'sdxl', bits: bitsOfNumber(0b101100111110110010010000011110111011000110011110n) },
  { scheme: 'flux', bits: bitsOfNumber(0b001010101111111010000111100111001111010100101110n) },
];

function chromaPlane(image: PixelImage, swapped: boolean): { plane: Float64Array; width: number; height: number } {
  const { width, channels, data } = image;
  const w = Math.floor(width / 4) * 4;
  const h = Math.floor(image.height / 4) * 4;
  const half = w / 2;
  const plane = new Float64Array(half * (h / 2));
  const u = new Float64Array(w * 2);
  for (let y = 0; y < h; y += 2) {
    for (let r = 0; r < 2; r++) {
      for (let x = 0; x < w; x++) {
        const p = ((y + r) * width + x) * channels;
        const red = data[swapped ? p + 2 : p];
        const blue = data[swapped ? p : p + 2];
        const luma = 0.299 * red + 0.587 * data[p + 1] + 0.114 * blue;
        const value = Math.round((blue - luma) * 0.492 + 128);
        u[r * w + x] = value < 0 ? 0 : value > 255 ? 255 : value;
      }
    }
    for (let x = 0; x < w; x += 2) plane[(y / 2) * half + x / 2] = (u[x] + u[x + 1] + u[w + x] + u[w + x + 1]) / 2;
  }
  return { plane, width: half, height: h / 2 };
}

function blockBits(plane: Float64Array, width: number, height: number): Uint8Array {
  const columns = Math.floor(width / BLOCK);
  const rows = Math.floor(height / BLOCK);
  const bits = new Uint8Array(rows * columns);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      let best = -1;
      let value = 0;
      for (let k = 1; k < BLOCK * BLOCK; k++) {
        const v = Math.abs(plane[(i * BLOCK + Math.floor(k / BLOCK)) * width + j * BLOCK + (k % BLOCK)]);
        if (v > best) {
          best = v;
          value = v;
        }
      }
      bits[i * columns + j] = value % SCALE > SCALE / 2 ? 1 : 0;
    }
  }
  return bits;
}

export function decodeBits(blocks: Uint8Array, length: number): Uint8Array {
  const sums = new Float64Array(length);
  const counts = new Float64Array(length);
  for (let n = 0; n < blocks.length; n++) {
    sums[n % length] += blocks[n];
    counts[n % length]++;
  }
  return Uint8Array.from(sums, (sum, i) => (counts[i] > 0 && (sum / counts[i]) * 255 > 127 ? 1 : 0));
}

export function detectInvisibleWatermark(image: PixelImage): WatermarkScheme | null {
  if (image.width * image.height < MIN_PIXELS) return null;
  for (const swapped of [false, true]) {
    const { plane, width, height } = chromaPlane(image, swapped);
    const blocks = blockBits(plane, width, height);
    for (const { scheme, bits } of PAYLOADS) {
      if (blocks.length < bits.length * 4) continue;
      const decoded = decodeBits(blocks, bits.length);
      let agreeing = 0;
      for (let i = 0; i < bits.length; i++) if (decoded[i] === bits[i]) agreeing++;
      if ((agreeing - bits.length / 2) / Math.sqrt(bits.length / 4) >= MIN_Z) return scheme;
    }
  }
  return null;
}
