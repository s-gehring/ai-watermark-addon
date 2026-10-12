import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { decodeBits, detectInvisibleWatermark } from '../../src/background/invisible-watermark';

async function load(name: string, jpegQuality?: number) {
  let image = sharp(new URL(`./fixtures/invisible-watermark/${name}.png`, import.meta.url).pathname);
  if (jpegQuality) image = sharp(await image.jpeg({ quality: jpegQuality }).toBuffer());
  const { data, info } = await image.removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, channels: 3 as const, data };
}

describe('detectInvisibleWatermark', () => {
  it('finds the Stable Diffusion watermark written by invisible-watermark', async () => {
    expect(detectInvisibleWatermark(await load('stable-diffusion'))).toBe('stable-diffusion');
  });

  it('finds the SDXL watermark written by invisible-watermark', async () => {
    expect(detectInvisibleWatermark(await load('sdxl'))).toBe('sdxl');
  });

  it('ignores an image without the watermark', async () => {
    expect(detectInvisibleWatermark(await load('clean'))).toBeNull();
  });

  it('does not find the watermark after JPEG compression', async () => {
    expect(detectInvisibleWatermark(await load('stable-diffusion', 90))).toBeNull();
  });

  it('skips images below 256 x 256 pixels', () => {
    expect(detectInvisibleWatermark({ width: 128, height: 128, channels: 3, data: new Uint8Array(128 * 128 * 3) })).toBeNull();
  });
});

describe('decodeBits', () => {
  it('takes the majority per bit position with the 127/255 cut-off', () => {
    expect(Array.from(decodeBits(Uint8Array.from([1, 0, 1, 1, 0, 1, 1, 0, 0]), 3))).toEqual([1, 0, 1]);
  });
});
