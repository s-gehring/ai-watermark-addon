import sharp, { type Sharp } from 'sharp';
import { describe, expect, it } from 'vitest';
import { isLossyWebp } from '../../src/background/format';

const pixels = (channels: 3 | 4) => sharp(Buffer.alloc(16 * 16 * channels, 128), { raw: { width: 16, height: 16, channels } });
const bytes = async (image: Sharp) => new Uint8Array(await image.toBuffer());

describe('isLossyWebp', () => {
  it('accepts opaque lossy WebP', async () => {
    expect(isLossyWebp(await bytes(pixels(3).webp({ quality: 80 })))).toBe(true);
  });

  it('rejects lossy WebP with transparency, lossless WebP and other formats', async () => {
    expect(isLossyWebp(await bytes(pixels(4).webp({ quality: 80 })))).toBe(false);
    expect(isLossyWebp(await bytes(pixels(3).webp({ lossless: true })))).toBe(false);
    expect(isLossyWebp(await bytes(pixels(3).jpeg()))).toBe(false);
    expect(isLossyWebp(await bytes(pixels(3).png()))).toBe(false);
    expect(isLossyWebp(new Uint8Array([0x52, 0x49, 0x46, 0x46]))).toBe(false);
  });
});
