import { describe, expect, it } from 'vitest';
import { TaskQueue } from '../../src/background/queue';
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

describe('TaskQueue', () => {
  it('runs tasks one at a time in order and survives failures', async () => {
    const queue = new TaskQueue();
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

  it('never runs more tasks at once than its concurrency allows', async () => {
    const queue = new TaskQueue(2);
    let running = 0;
    let peak = 0;
    const task = (ms: number) => async () => {
      peak = Math.max(peak, ++running);
      await new Promise((resolve) => setTimeout(resolve, ms));
      running--;
    };
    await Promise.all([5, 1, 3, 1, 2].map((ms) => queue.run(task(ms))));
    expect(peak).toBe(2);
  });
});
