import { describe, expect, it } from 'vitest';
import { checkedCount, isFlagged, isPossible, progressOf, sumStats, type Verdict } from '../../src/shared/messages';

describe('frame stats', () => {
  it('sums all frames of a tab', () => {
    const total = sumStats([
      { flagged: 1, possible: 1, clean: 2, pending: 3, failed: 0 },
      { flagged: 0, possible: 2, clean: 1, pending: 1, failed: 1 },
    ]);
    expect(total).toEqual({ flagged: 1, possible: 3, clean: 3, pending: 4, failed: 1 });
    expect(checkedCount(total)).toBe(8);
  });

  it('reports progress only while checks are pending', () => {
    expect(progressOf({ flagged: 1, possible: 0, clean: 2, pending: 1, failed: 0 })).toBe(0.75);
    expect(progressOf({ flagged: 0, possible: 1, clean: 0, pending: 3, failed: 0 })).toBe(0.25);
    expect(progressOf({ flagged: 0, possible: 0, clean: 0, pending: 5, failed: 0 })).toBe(0);
    expect(progressOf({ flagged: 1, possible: 0, clean: 2, pending: 0, failed: 1 })).toBeNull();
  });
});

describe('isPossible', () => {
  const scored = (score: number): Verdict => ({ ok: true, score, source: 'model', models: [score], metadata: 'clear', watermark: 'clear' });

  it('covers model scores from the lower bound up to, but not including, the threshold', () => {
    expect(isPossible(scored(0.79), 0.95, 0.8)).toBe(false);
    expect(isPossible(scored(0.8), 0.95, 0.8)).toBe(true);
    expect(isPossible(scored(0.949), 0.95, 0.8)).toBe(true);
    expect(isPossible(scored(0.95), 0.95, 0.8)).toBe(false);
  });

  it('never applies to platform, metadata or watermark hits and failed checks', () => {
    expect(isPossible({ ok: true, score: 1, source: 'platform', platform: 'pixiv' }, 0.99, 0.8)).toBe(false);
    expect(isPossible({ ok: true, score: 1, source: 'metadata', reason: 'ai-generated' }, 0.95, 0.8)).toBe(false);
    expect(isPossible({ ok: true, score: 1, source: 'synthid', provider: 'google', metadata: 'clear' }, 0.99, 0.8)).toBe(false);
    expect(isPossible({ ok: true, score: 1, source: 'watermark', scheme: 'flux', metadata: 'clear' }, 0.99, 0.8)).toBe(false);
    expect(isPossible({ ok: false, error: 'model' }, 0.95, 0.8)).toBe(false);
  });

  it('leaves a platform label a certain hit at every threshold', () => {
    expect(isFlagged({ ok: true, score: 1, source: 'platform', platform: 'commons' }, 0.99)).toBe(true);
  });

  it('is off without a lower bound', () => {
    expect(isPossible(scored(0.9), 0.95, null)).toBe(false);
  });
});
