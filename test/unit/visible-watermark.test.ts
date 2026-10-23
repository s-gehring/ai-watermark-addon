import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { dalleBarScore, detectVisibleWatermark, geminiLogoScore, klingLogoScore, visibleMarkScores } from '../../src/background/visible-watermark';

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

  it.each([
    ['doubao', 'doubao-visible'],
    ['jimeng', 'jimeng-visible'],
    ['qwen', 'qwen-visible'],
    ['kling-cjk', 'kling-visible'],
    ['grok-2024', 'grok-visible'],
    ['grok-aurora', 'grok-visible'],
    ['grok', 'grok-visible'],
    ['bing', 'bing-visible'],
    ['meta', 'meta-visible'],
    ['dalle2', 'dalle-visible'],
  ])('finds the %s mark', async (name, scheme) => {
    expect(detectVisibleWatermark(await load(`visible-watermark/${name}.jpg`))).toBe(scheme);
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
    expect(dalleBarScore(tiny)).toBe(-1);
    expect(Math.max(...Object.values(visibleMarkScores(tiny)))).toBe(-1);
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

describe('visibleMarkScores', () => {
  it.each([
    ['doubao', 0.93336],
    ['jimeng', 0.93843],
    ['qwen', 0.93284],
    ['kling-cjk', 0.68394],
    ['grok-2024', 0.92458],
    ['grok-aurora', 0.92138],
    ['grok', 0.9398],
    ['bing', 0.98068],
    ['meta', 0.81228],
  ] as const)('matches the Python reference implementation for %s', async (name, score) => {
    expect(visibleMarkScores(await load(`visible-watermark/${name}.jpg`))[name]).toBeCloseTo(score, 4);
  });

  it('matches the Python reference implementation without a mark', async () => {
    const scores = visibleMarkScores(await load('visible-watermark/clean.jpg'));
    expect(scores.doubao).toBeCloseTo(-0.03757, 4);
    expect(scores.jimeng).toBeCloseTo(0.21871, 4);
    expect(scores.qwen).toBeCloseTo(0.12114, 4);
    expect(scores['kling-cjk']).toBeCloseTo(-0.03224, 4);
    expect(scores['grok-2024']).toBeCloseTo(0.22446, 4);
    expect(scores['grok-aurora']).toBeCloseTo(0.23728, 4);
    expect(scores.grok).toBeCloseTo(0.04992, 4);
    expect(scores.bing).toBeCloseTo(0.32185, 4);
    expect(scores.meta).toBeCloseTo(-0.02684, 4);
  });
});

describe('dalleBarScore', () => {
  it('matches the Python reference implementation', async () => {
    expect(dalleBarScore(await load('visible-watermark/dalle2.jpg'))).toBeCloseTo(0.80063, 4);
    expect(dalleBarScore(await load('visible-watermark/clean.jpg'))).toBe(-1);
    expect(dalleBarScore(await load('visible-watermark/bing.jpg'))).toBe(-1);
  });
});
