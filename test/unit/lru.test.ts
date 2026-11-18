import { describe, expect, it } from 'vitest';
import { firstResult } from '../../src/background/queue';
import { LruCache } from '../../src/shared/lru';

describe('LruCache', () => {
  it('evicts the least recently used entry', () => {
    const cache = new LruCache<string, number>(2);
    cache.set('a', 1);
    cache.set('b', 2);
    cache.get('a');
    cache.set('c', 3);
    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('a')).toBe(1);
    expect(cache.get('c')).toBe(3);
  });
});

describe('firstResult', () => {
  it('returns the first candidate that loads and stops there', async () => {
    const asked: string[] = [];
    const load = async (candidate: string) => {
      asked.push(candidate);
      return candidate === 'b' || candidate === 'c' ? candidate.toUpperCase() : null;
    };
    expect(await firstResult(['a', 'b', 'c'], load)).toBe('B');
    expect(asked).toEqual(['a', 'b']);
  });

  it('returns null when no candidate loads', async () => {
    expect(await firstResult(['a', 'b'], async () => null)).toBeNull();
    expect(await firstResult([], async () => 'x')).toBeNull();
  });
});
