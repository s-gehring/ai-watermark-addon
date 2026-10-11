import { describe, expect, it } from 'vitest';
import { checkedCount, progressOf, sumStats } from '../../src/shared/messages';

describe('frame stats', () => {
  it('sums all frames of a tab', () => {
    const total = sumStats([
      { flagged: 1, clean: 2, pending: 3, failed: 0 },
      { flagged: 0, clean: 1, pending: 1, failed: 1 },
    ]);
    expect(total).toEqual({ flagged: 1, clean: 3, pending: 4, failed: 1 });
    expect(checkedCount(total)).toBe(5);
  });

  it('reports progress only while checks are pending', () => {
    expect(progressOf({ flagged: 1, clean: 2, pending: 1, failed: 0 })).toBe(0.75);
    expect(progressOf({ flagged: 0, clean: 0, pending: 5, failed: 0 })).toBe(0);
    expect(progressOf({ flagged: 1, clean: 2, pending: 0, failed: 1 })).toBeNull();
  });
});
