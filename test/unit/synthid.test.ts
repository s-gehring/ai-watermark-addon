import { describe, expect, it } from 'vitest';
import {
  cropHypotheses,
  cropSearchScores,
  detectSynthId,
  fft,
  searchCroppedSynthId,
  type CropSearchConfig,
  localContrast,
  resampledPlanes,
  rfft2,
  synthIdScore,
  type SynthIdDetector,
  type SynthIdTemplate,
} from '../../src/background/synthid';
import type { PixelImage } from '../../src/background/preprocess';
import type { SynthIdProvider } from '../../src/shared/messages';

function patternedImage(width: number, height: number): PixelImage {
  const data = new Float64Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) data[(y * width + x) * 3 + c] = (x * 7 + y * 13 + c * 29 + ((x * y) % 17)) % 256;
    }
  }
  return { width, height, channels: 3, data };
}

function arithmeticTemplate(size: number): SynthIdTemplate {
  const count = 3 * size * (size / 2 + 1);
  const weights = new Uint8Array(count);
  const phases = new Uint8Array(count);
  for (let t = 0; t < count; t++) {
    weights[t] = t % 11 === 0 ? (t * 13) % 256 : 0;
    phases[t] = (t * 97) % 256;
  }
  return { size, weightScale: 0.01, weights, phases };
}

function random(seed: number): () => number {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

const CARRIERS: Array<[number, number]> = (() => {
  const next = random(3);
  const picked = new Map<string, [number, number]>();
  while (picked.size < 40) {
    const ky = Math.floor(next() * 49) - 24;
    const kx = 4 + Math.floor(next() * 21);
    if (ky !== 0) picked.set(`${ky},${kx}`, [ky, kx]);
  }
  return [...picked.values()];
})();

function carrierImage(width: number, height: number, amplitude: number, noise: number): PixelImage {
  const next = random(7);
  const offsets = CARRIERS.map(() => next() * 2 * Math.PI);
  const data = new Float64Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let wave = 0;
      CARRIERS.forEach(([ky, kx], i) => {
        wave += Math.cos(2 * Math.PI * ((ky * (y + 0.5)) / height + (kx * (x + 0.5)) / width) + offsets[i]);
      });
      for (let c = 0; c < 3; c++) {
        data[(y * width + x) * 3 + c] = 128 + (c === 0 ? amplitude * wave : 0) + noise * (next() - 0.5);
      }
    }
  }
  return { width, height, channels: 3, data };
}

function templateFrom(image: PixelImage, size: number): SynthIdTemplate {
  const half = size / 2 + 1;
  const weights = new Uint8Array(3 * size * half);
  const phases = new Uint8Array(3 * size * half);
  const { re, im } = rfft2(localContrast(resampledPlanes(image, size)[2], size), size);
  for (const [ky, kx] of CARRIERS) {
    const row = (ky + size) % size;
    const t = (2 * size + row) * half + kx;
    weights[t] = 255;
    phases[t] = Math.round((Math.atan2(im[row * half + kx], re[row * half + kx]) / (2 * Math.PI)) * 256) & 255;
  }
  return { size, weightScale: 1 / 255, weights, phases };
}

describe('fft', () => {
  it('matches a direct DFT', () => {
    const next = random(1);
    const input = Array.from({ length: 16 }, () => next() - 0.5);
    const re = Float64Array.from(input);
    const im = new Float64Array(16);
    fft(re, im);
    for (let k = 0; k < 16; k++) {
      let sumRe = 0;
      let sumIm = 0;
      input.forEach((v, n) => {
        sumRe += v * Math.cos((-2 * Math.PI * k * n) / 16);
        sumIm += v * Math.sin((-2 * Math.PI * k * n) / 16);
      });
      expect(re[k]).toBeCloseTo(sumRe, 10);
      expect(im[k]).toBeCloseTo(sumIm, 10);
    }
  });
});

describe('synthIdScore', () => {
  it.each([
    [64, 150, 97, -0.31185490612821987],
    [128, 100, 90, 0.6851699121893623],
    [64, 300, 64, -0.16965899414319735],
  ])('matches the Python reference (template %i, image %ix%i)', (size, width, height, expected) => {
    expect(synthIdScore(patternedImage(width, height), arithmeticTemplate(size))).toBeCloseTo(expected, 5);
  });

  it('finds carriers defined in cycles per image at any resolution', () => {
    const template = templateFrom(carrierImage(192, 128, 2, 0), 64);
    expect(synthIdScore(carrierImage(384, 256, 2, 40), template)).toBeGreaterThan(5);
    expect(synthIdScore(carrierImage(150, 100, 2, 40), template)).toBeGreaterThan(5);
    expect(Math.abs(synthIdScore(carrierImage(384, 256, 0, 40), template))).toBeLessThan(3);
  });

  it('skips color channels without variation for templates that set a minimum', () => {
    const chromaOnly = { ...arithmeticTemplate(64), minChannelStd: 0.5 };
    chromaOnly.weights.fill(0, 0, 64 * 33);
    const colored = patternedImage(150, 97);
    const gray = { ...colored, data: Float64Array.from(colored.data, (_, i) => colored.data[i - (i % 3)]) };
    expect(synthIdScore(gray, chromaOnly)).toBe(0);
    expect(synthIdScore(colored, chromaOnly)).not.toBe(0);
  });

  it('skips images that are too small to carry the pattern', () => {
    expect(synthIdScore(patternedImage(48, 300), arithmeticTemplate(64))).toBe(0);
  });
});

describe('detectSynthId', () => {
  const pattern = carrierImage(192, 128, 2, 0);
  const detector = (provider: SynthIdProvider, size: number): SynthIdDetector => ({
    ...templateFrom(pattern, size),
    provider,
    threshold: 5,
  });

  it('reports the provider of the first matching template', () => {
    expect(detectSynthId(carrierImage(384, 256, 2, 40), [detector('openai', 64), detector('google', 128)])).toBe('openai');
    expect(detectSynthId(carrierImage(384, 256, 2, 40), [detector('google', 128), detector('openai', 64)])).toBe('google');
  });

  it('runs prescreen templates only once the same provider scored at least 3', () => {
    const image = carrierImage(384, 256, 2, 40);
    const gate = (provider: SynthIdProvider) => ({ ...detector(provider, 64), threshold: Infinity });
    const fine = { ...detector('openai', 128), prescreen: true };
    expect(detectSynthId(image, [gate('openai'), fine])).toBe('openai');
    expect(detectSynthId(image, [gate('google'), fine])).toBeNull();
    expect(detectSynthId(image, [fine])).toBeNull();
  });

  it('finds nothing in images without the pattern', () => {
    expect(detectSynthId(carrierImage(384, 256, 0, 40), [detector('openai', 64), detector('google', 128)])).toBeNull();
  });
});

describe('crop search', () => {
  const limits = { minAxis: 0.3, minBoth: 0.75, maxSkew: 0.02 };

  it('derives scale hypotheses from the output sizes', () => {
    expect(cropHypotheses(768, 768, [[1408, 768], [1024, 1024]], limits)).toEqual([
      { sy: 1, sx: 768 / 1408, mode: 'x' },
      { sy: 0.75, sx: 0.75, mode: 'xy' },
    ]);
    expect(cropHypotheses(973, 973, [[1024, 1024]], limits)).toEqual([{ sy: 973 / 1024, sx: 973 / 1024, mode: 'xy' }]);
    expect(cropHypotheses(1024, 1024, [[1024, 1024]], limits)).toEqual([]);
    expect(cropHypotheses(900, 1000, [[1024, 1024]], limits)).toEqual([{ sy: 1, sx: 0.9, mode: 'x' }]);
  });

  it.each([
    [300, 250, 3.325327784293236],
    [200, 300, 3.617012283024607],
    [390, 290, 3.251783138379988],
    [304, 247, 3.2874127584728967],
  ])('matches the Python reference (image %ix%i)', (width, height, expected) => {
    const config: CropSearchConfig = { ...limits, threshold: 8, sizes: { google: [[400, 300], [300, 400], [320, 260]] } };
    const detector: SynthIdDetector = { ...arithmeticTemplate(512), provider: 'google', threshold: 10, cropSearch: true };
    expect(cropSearchScores(patternedImage(width, height), [detector], config).google).toBeCloseTo(expected, 4);
  });

  const original = carrierImage(384, 256, 2, 40);
  const detector: SynthIdDetector = { ...templateFrom(carrierImage(384, 256, 2, 0), 512), provider: 'openai', threshold: 5, cropSearch: true };
  const config: CropSearchConfig = { ...limits, threshold: 5, sizes: { openai: [[384, 256]] } };
  const crop = (image: PixelImage, left: number, top: number, width: number, height: number): PixelImage => {
    const data = new Float64Array(width * height * 3);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let c = 0; c < 3; c++) data[(y * width + x) * 3 + c] = image.data[((y + top) * image.width + x + left) * 3 + c];
      }
    }
    return { width, height, channels: 3, data };
  };

  it.each([
    ['one full axis', 64, 0, 256, 256],
    ['a uniform border', 10, 7, 365, 243],
  ])('finds the pattern in a crop with %s that the regular check misses', (_, left, top, width, height) => {
    const cropped = crop(original, left, top, width, height);
    expect(detectSynthId(cropped, [detector])).toBeNull();
    expect(searchCroppedSynthId(cropped, [detector], config)).toBe('openai');
  });

  it('finds nothing in crops without the pattern', () => {
    const cropped = crop(carrierImage(384, 256, 0, 40), 64, 0, 256, 256);
    expect(searchCroppedSynthId(cropped, [detector], config)).toBeNull();
  });

  it('only uses templates marked for the crop search', () => {
    const cropped = crop(original, 64, 0, 256, 256);
    expect(searchCroppedSynthId(cropped, [{ ...detector, cropSearch: false }], config)).toBeNull();
  });
});
