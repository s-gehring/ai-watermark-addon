import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { detectVisibleWatermark, geminiLogoScore, klingLogoScore } from '../../src/background/visible-watermark';

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

  it('finds the older Kling logo, scaled with the long side', async () => {
    expect(detectVisibleWatermark(await load('visible-watermark/kling-2024.jpg'))).toBe('kling-visible');
  });

  it('finds the current Kling logo, scaled with the short side', async () => {
    expect(detectVisibleWatermark(await load('visible-watermark/kling-2026.jpg'))).toBe('kling-visible');
  });

  it('ignores the same image without the logo', async () => {
    expect(detectVisibleWatermark(await load('visible-watermark/clean.jpg'))).toBeNull();
  });

  it('ignores a bright image without the logo', async () => {
    expect(detectVisibleWatermark(await load('invisible-watermark/clean.png'))).toBeNull();
  });

  it('skips images too small to carry the logo', () => {
    const tiny = { width: 40, height: 40, channels: 3 as const, data: new Uint8Array(40 * 40 * 3) };
    expect(geminiLogoScore(tiny)).toBe(-1);
    expect(klingLogoScore(tiny)).toBe(-1);
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

describe('klingLogoScore', () => {
  it('matches the Python reference implementation', async () => {
    expect(klingLogoScore(await load('visible-watermark/kling-2024.jpg'))).toBeCloseTo(0.89247, 4);
    expect(klingLogoScore(await load('visible-watermark/kling-2026.jpg'))).toBeCloseTo(0.84979, 4);
    expect(klingLogoScore(await load('visible-watermark/clean.jpg'))).toBeCloseTo(0.02457, 4);
    expect(klingLogoScore(await load('invisible-watermark/clean.png'))).toBeCloseTo(0.01947, 4);
  });
});
