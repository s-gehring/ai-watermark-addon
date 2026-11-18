import { describe, expect, it } from 'vitest';
import { mayContinue, outOfView } from '../../src/content/scanner';

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

describe('outOfView', () => {
  it('lists the open requests whose image no element in view shows', () => {
    const awaited = new Map([
      ['https://a.test/1.jpg', 1],
      ['https://a.test/2.jpg', 2],
      ['https://a.test/3.jpg', 3],
    ]);
    expect(outOfView(awaited, ['https://a.test/2.jpg', undefined, 'https://a.test/other.jpg', 'https://a.test/2.jpg'])).toEqual([1, 3]);
    expect(outOfView(awaited, [])).toEqual([1, 2, 3]);
    expect(outOfView(new Map(), ['https://a.test/1.jpg'])).toEqual([]);
  });
});
