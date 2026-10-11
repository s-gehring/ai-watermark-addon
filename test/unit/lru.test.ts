import { describe, expect, it } from 'vitest';
import { SerialQueue } from '../../src/background/queue';
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

describe('SerialQueue', () => {
  it('runs tasks one at a time in order and survives failures', async () => {
    const queue = new SerialQueue();
    const log: string[] = [];
    const task = (name: string, ms: number, fail = false) => () =>
      new Promise<string>((resolve, reject) =>
        setTimeout(() => {
          log.push(name);
          fail ? reject(new Error(name)) : resolve(name);
        }, ms),
      );
    const results = await Promise.allSettled([
      queue.run(task('slow', 20)),
      queue.run(task('broken', 1, true)),
      queue.run(task('fast', 1)),
    ]);
    expect(log).toEqual(['slow', 'broken', 'fast']);
    expect(results.map((r) => r.status)).toEqual(['fulfilled', 'rejected', 'fulfilled']);
  });
});
