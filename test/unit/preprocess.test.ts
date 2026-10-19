import { describe, expect, it } from 'vitest';
import { extraViews, inputShape, isScorable, resizedSize, toModelInput, type InputSpec } from '../../src/background/preprocess';
import cases from './fixtures/pil-bicubic.json';
import bilinearCases from './fixtures/pil-bilinear.json';

const toHwc = (out: Float32Array, plane: number) =>
  Array.from({ length: plane * 3 }, (_, i) => Math.round(out[(i % 3) * plane + Math.floor(i / 3)] * 255));

describe('toModelInput', () => {
  it.each(cases.map((c) => [`${c.width}x${c.height} -> ${c.shortestEdge}/${c.crop}`, c] as const))(
    'matches Pillow bicubic resize and center crop for %s',
    (_, c) => {
      const out = toModelInput(
        { width: c.width, height: c.height, channels: 3, data: c.pixels },
        { shortestEdge: c.shortestEdge, crop: c.crop, mean: [0, 0, 0], std: [1, 1, 1] },
      );
      expect(toHwc(out, c.crop * c.crop)).toEqual(c.expected);
    },
  );

  it.each(bilinearCases.map((c) => [`${c.width}x${c.height} -> ${c.size.join('x')}`, c] as const))(
    'matches Pillow bilinear resize to a fixed size for %s',
    (_, c) => {
      const size = c.size as [number, number];
      const out = toModelInput(
        { width: c.width, height: c.height, channels: 3, data: c.pixels },
        { size, resample: 'bilinear', mean: [0, 0, 0], std: [1, 1, 1] },
      );
      expect(toHwc(out, size[0] * size[1])).toEqual(c.expected);
    },
  );

  it('reads RGBA input and normalizes per channel', () => {
    const data = new Uint8ClampedArray(4 * 4 * 4).fill(255);
    const out = toModelInput(
      { width: 4, height: 4, channels: 4, data },
      { shortestEdge: 4, crop: 2, mean: [0.5, 0.5, 0.5], std: [0.25, 0.5, 1] },
    );
    expect(Array.from(out)).toEqual([2, 2, 2, 2, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]);
  });
});

describe('extraViews', () => {
  const spec: InputSpec = { shortestEdge: 440, crop: 384, resample: 'bilinear', mean: [0, 0, 0], std: [1, 1, 1] };
  const image = (width: number, height: number) => ({
    width,
    height,
    channels: 3 as const,
    data: Uint8Array.from({ length: width * height * 3 }, (_, i) => (i * 7) % 256),
  });

  it('adds one center crop per distinct shorter edge, capped at the native size', () => {
    expect(extraViews(image(1280, 853), spec, [576, 768])).toHaveLength(2);
    expect(extraViews(image(800, 533), spec, [576, 768])).toHaveLength(1);
    expect(extraViews(image(440, 600), spec, [576, 768])).toHaveLength(0);
  });

  it('resizes larger images to each shorter edge before the center crop', () => {
    const img = image(1280, 853);
    const views = extraViews(img, spec, [576, 768]);
    expect(views[0]).toEqual(toModelInput(img, { ...spec, shortestEdge: 576 }));
    expect(views[1]).toEqual(toModelInput(img, { ...spec, shortestEdge: 768 }));
  });

  it('crops natively without resampling when the image is smaller than the edge', () => {
    const img = image(500, 400);
    const [view] = extraViews(img, spec, [576]);
    const plane = 384 * 384;
    expect(view).toHaveLength(3 * plane);
    expect(view[0]).toBeCloseTo(img.data[(8 * 500 + 58) * 3] / 255, 6);
  });

  it('upscales images below the crop size to exactly the crop size', () => {
    const [view] = extraViews(image(800, 365), spec, [576]);
    expect(view.every(Number.isFinite)).toBe(true);
  });
});

describe('inputShape', () => {
  it('returns width and height of the model input', () => {
    expect(inputShape({ shortestEdge: 440, crop: 384, mean: [0, 0, 0], std: [1, 1, 1] })).toEqual([384, 384]);
    expect(inputShape({ size: [224, 160], mean: [0, 0, 0], std: [1, 1, 1] })).toEqual([224, 160]);
  });
});

describe('resizedSize', () => {
  it('scales the shorter edge with the float truncation of the reference pipeline', () => {
    expect(resizedSize(1000, 750, 440)).toEqual([586, 440]);
    expect(resizedSize(300, 900, 440)).toEqual([439, 1320]);
  });
});

describe('isScorable', () => {
  it('rejects tiny and extremely elongated images', () => {
    expect(isScorable({ width: 640, height: 480 })).toBe(true);
    expect(isScorable({ width: 31, height: 400 })).toBe(false);
    expect(isScorable({ width: 2000, height: 200 })).toBe(false);
  });
});
