import { describe, expect, it } from 'vitest';
import { mayContinue } from '../../src/content/scanner';

const deadline = (remaining: number, didTimeout: boolean): IdleDeadline => ({ didTimeout, timeRemaining: () => remaining });

describe('mayContinue', () => {
  it('works through the first chunk without waiting for idle time', () => {
    expect(mayContinue(undefined)).toBe(true);
  });

  it('follows the idle time the browser grants', () => {
    expect(mayContinue(deadline(12, false))).toBe(true);
    expect(mayContinue(deadline(0.5, false))).toBe(false);
  });

  it('keeps discovering on a page that never goes idle, once the idle request times out', () => {
    expect(mayContinue(deadline(0, true))).toBe(true);
  });
});
