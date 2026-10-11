export interface PixelImage {
  width: number;
  height: number;
  channels: 3 | 4;
  data: ArrayLike<number>;
}

export interface InputSpec {
  shortestEdge: number;
  crop: number;
  mean: [number, number, number];
  std: [number, number, number];
}

const PRECISION_BITS = 22;
const SUPPORT = 2;

function bicubic(x: number): number {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

interface Coefficients {
  ksize: number;
  bounds: Int32Array;
  weights: Int32Array;
}

function coefficients(inSize: number, outSize: number): Coefficients {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = SUPPORT * filterScale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const weights = new Int32Array(outSize * ksize);
  const k = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    let sum = 0;
    for (let x = 0; x < xmax; x++) {
      k[x] = bicubic((x + xmin - center + 0.5) / filterScale);
      sum += k[x];
    }
    for (let x = 0; x < ksize; x++) {
      const w = x < xmax && sum !== 0 ? k[x] / sum : 0;
      weights[xx * ksize + x] = Math.trunc(w < 0 ? -0.5 + w * (1 << PRECISION_BITS) : 0.5 + w * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, weights };
}

function clip8(sum: number): number {
  const v = (sum + (1 << (PRECISION_BITS - 1))) >> PRECISION_BITS;
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

export function isScorable(image: Pick<PixelImage, 'width' | 'height'>): boolean {
  const short = Math.min(image.width, image.height);
  const long = Math.max(image.width, image.height);
  return short >= 32 && long / short <= 8;
}

export function resizedSize(width: number, height: number, shortestEdge: number): [number, number] {
  const scale = shortestEdge / Math.min(width, height);
  return [Math.trunc(width * scale), Math.trunc(height * scale)];
}

export function toModelInput(image: PixelImage, spec: InputSpec): Float32Array {
  const { width, height, channels, data } = image;
  const [outW, outH] = resizedSize(width, height, spec.shortestEdge);
  const crop = spec.crop;
  const left = Math.floor((outW - crop) / 2);
  const top = Math.floor((outH - crop) / 2);

  const horizontal = coefficients(width, outW);
  const vertical = coefficients(height, outH);
  const firstRow = vertical.bounds[top * 2];
  const lastRow = vertical.bounds[(top + crop - 1) * 2] + vertical.bounds[(top + crop - 1) * 2 + 1];

  const rows = lastRow - firstRow;
  const temp = new Uint8Array(rows * crop * 3);
  for (let y = 0; y < rows; y++) {
    const rowOffset = (y + firstRow) * width;
    for (let xx = 0; xx < crop; xx++) {
      const ox = xx + left;
      const xmin = horizontal.bounds[ox * 2];
      const xmax = horizontal.bounds[ox * 2 + 1];
      const kOffset = ox * horizontal.ksize;
      let r = 0;
      let g = 0;
      let b = 0;
      for (let x = 0; x < xmax; x++) {
        const w = horizontal.weights[kOffset + x];
        const p = (rowOffset + xmin + x) * channels;
        r += data[p] * w;
        g += data[p + 1] * w;
        b += data[p + 2] * w;
      }
      const t = (y * crop + xx) * 3;
      temp[t] = clip8(r);
      temp[t + 1] = clip8(g);
      temp[t + 2] = clip8(b);
    }
  }

  const plane = crop * crop;
  const out = new Float32Array(3 * plane);
  for (let yy = 0; yy < crop; yy++) {
    const oy = yy + top;
    const ymin = vertical.bounds[oy * 2] - firstRow;
    const ymax = vertical.bounds[oy * 2 + 1];
    const kOffset = oy * vertical.ksize;
    for (let xx = 0; xx < crop; xx++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let y = 0; y < ymax; y++) {
        const w = vertical.weights[kOffset + y];
        const t = ((ymin + y) * crop + xx) * 3;
        r += temp[t] * w;
        g += temp[t + 1] * w;
        b += temp[t + 2] * w;
      }
      const i = yy * crop + xx;
      out[i] = (clip8(r) / 255 - spec.mean[0]) / spec.std[0];
      out[plane + i] = (clip8(g) / 255 - spec.mean[1]) / spec.std[1];
      out[2 * plane + i] = (clip8(b) / 255 - spec.mean[2]) / spec.std[2];
    }
  }
  return out;
}
