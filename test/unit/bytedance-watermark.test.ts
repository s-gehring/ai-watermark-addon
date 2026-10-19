import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { bytedanceTileScore, detectBytedanceWatermark } from '../../src/background/bytedance-watermark';

async function load(name: string) {
  const path = new URL(`./fixtures/bytedance-watermark/${name}`, import.meta.url).pathname;
  const { data, info } = await sharp(path).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, channels: 3 as const, data };
}

describe('detectBytedanceWatermark', () => {
  it('finds the tiled chroma pattern anchored at the top left corner', async () => {
    expect(detectBytedanceWatermark(await load('tile.jpg'))).toBe('bytedance');
  });

  it('ignores the same image without the pattern', async () => {
    expect(detectBytedanceWatermark(await load('clean.jpg'))).toBeNull();
  });

  it('ignores the pattern once the image is shifted by half a tile', async () => {
    const image = await load('tile.jpg');
    const shift = (48 * image.width + 56) * 3;
    const shifted = { ...image, height: image.height - 48, data: image.data.subarray(shift) };
    expect(detectBytedanceWatermark(shifted)).toBeNull();
  });

  it('skips images with too few tiles', () => {
    const small = { width: 300, height: 300, channels: 3 as const, data: new Uint8Array(300 * 300 * 3) };
    expect(bytedanceTileScore(small)).toBeNull();
  });
});
