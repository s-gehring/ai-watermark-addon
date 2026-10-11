import { describe, expect, it } from 'vitest';
import { isScorable, resizedSize, toModelInput } from '../../src/background/preprocess';
import cases from './fixtures/pil-bicubic.json';

describe('toModelInput', () => {
  it.each(cases.map((c) => [`${c.width}x${c.height} -> ${c.shortestEdge}/${c.crop}`, c] as const))(
    'matches Pillow bicubic resize and center crop for %s',
    (_, c) => {
      const out = toModelInput(
        { width: c.width, height: c.height, channels: 3, data: c.pixels },
        { shortestEdge: c.shortestEdge, crop: c.crop, mean: [0, 0, 0], std: [1, 1, 1] },
      );
      const plane = c.crop * c.crop;
      const hwc = Array.from({ length: plane * 3 }, (_, i) => {
        const pixel = Math.floor(i / 3);
        return Math.round(out[(i % 3) * plane + pixel] * 255);
      });
      expect(hwc).toEqual(c.expected);
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
