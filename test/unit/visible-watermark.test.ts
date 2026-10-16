import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { detectVisibleWatermark, geminiLogoScore } from '../../src/background/visible-watermark';

async function load(name: string) {
  const path = new URL(`./fixtures/${name}`, import.meta.url).pathname;
  const { data, info } = await sharp(path).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, channels: 3 as const, data };
}

describe('detectVisibleWatermark', () => {
  it('finds the Gemini logo with the small margin of the older layout', async () => {
    expect(detectVisibleWatermark(await load('visible-watermark/gemini-v1.jpg'))).toBe('gemini-visible');
  });

  it('finds the Gemini logo with the wide margin of the current layout', async () => {
    expect(detectVisibleWatermark(await load('visible-watermark/gemini-v2.jpg'))).toBe('gemini-visible');
  });

  it('ignores the same image without the logo', async () => {
    expect(detectVisibleWatermark(await load('visible-watermark/clean.jpg'))).toBeNull();
  });

  it('ignores a bright image without the logo', async () => {
    expect(detectVisibleWatermark(await load('invisible-watermark/clean.png'))).toBeNull();
  });

  it('skips images too small to carry the logo', () => {
    expect(geminiLogoScore({ width: 40, height: 40, channels: 3, data: new Uint8Array(40 * 40 * 3) })).toBe(-1);
  });
});

describe('geminiLogoScore', () => {
  it('matches the Python reference implementation', async () => {
    expect(geminiLogoScore(await load('visible-watermark/gemini-v1.jpg'))).toBeCloseTo(0.98132, 4);
    expect(geminiLogoScore(await load('visible-watermark/gemini-v2.jpg'))).toBeCloseTo(0.97237, 4);
    expect(geminiLogoScore(await load('visible-watermark/clean.jpg'))).toBeCloseTo(0.56096, 4);
    expect(geminiLogoScore(await load('invisible-watermark/clean.png'))).toBeCloseTo(0.63297, 4);
  });
});
