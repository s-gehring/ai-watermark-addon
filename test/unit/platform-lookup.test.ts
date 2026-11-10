import { describe, expect, it, vi } from 'vitest';
import { PlatformLookup, type LabelFetcher } from '../../src/background/platform-lookup';

const always = () => true;
const ids = (count: number, from = 1) => Array.from({ length: count }, (_, i) => String(from + i));

function answering(labeled: string[], truncated = false) {
  return vi.fn<LabelFetcher>(async (_platform, keys) => ({ labels: new Map(keys.map((key) => [key, labeled.includes(key)])), truncated }));
}

describe('PlatformLookup', () => {
  it('answers for every key and asks only once per key', async () => {
    const load = answering(['2']);
    const lookup = new PlatformLookup(load);
    expect(await lookup.labels('pixiv', ['1', '2', '2'], always)).toEqual({ 1: false, 2: true });
    expect(await lookup.labels('pixiv', ['1', '2'], always)).toEqual({ 1: false, 2: true });
    expect(load).toHaveBeenCalledTimes(1);
    expect(load).toHaveBeenCalledWith('pixiv', ['1', '2']);
  });

  it('splits many keys into batches', async () => {
    const load = answering([]);
    await new PlatformLookup(load).labels('pixiv', ids(120), always);
    expect(load.mock.calls.map(([, keys]) => keys.length)).toEqual([50, 50, 20]);
  });

  it('leaves keys unanswered when the page view has no requests left', async () => {
    const load = answering(['1', '60']);
    let left = 1;
    const lookup = new PlatformLookup(load);
    const answers = await lookup.labels('pixiv', ids(120), () => left-- > 0);
    expect(Object.keys(answers)).toHaveLength(50);
    expect(answers['1']).toBe(true);
    expect(load).toHaveBeenCalledTimes(1);
    expect(await lookup.labels('pixiv', ['60'], always)).toEqual({ 60: true });
  });

  it('leaves keys unanswered after a failed request and asks again', async () => {
    const load = vi.fn<LabelFetcher>(async () => null);
    const lookup = new PlatformLookup(load);
    expect(await lookup.labels('commons', ['File:A.jpg'], always)).toEqual({});
    expect(await lookup.labels('commons', ['File:A.jpg'], always)).toEqual({});
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('does not take a work that pixiv leaves out of its answer for unlabeled', async () => {
    const load = vi.fn<LabelFetcher>(async () => ({ labels: new Map([['1', false]]), truncated: false }));
    const lookup = new PlatformLookup(load);
    expect(await lookup.labels('pixiv', ['1', '2'], always)).toEqual({ 1: false });
    expect(await lookup.labels('pixiv', ['1', '2'], always)).toEqual({ 1: false });
    expect(load.mock.calls.map(([, keys]) => keys)).toEqual([['1', '2'], ['2']]);
  });

  it('asks again for the files a truncated answer left out, within the requests of the page view', async () => {
    const answers = [['File:A.jpg'], ['File:B.jpg'], ['File:C.jpg']];
    const load = vi.fn<LabelFetcher>(async () => ({ labels: new Map(answers.shift()!.map((key) => [key, key === 'File:B.jpg'])), truncated: true }));
    let left = 2;
    const lookup = new PlatformLookup(load);
    expect(await lookup.labels('commons', ['File:A.jpg', 'File:B.jpg', 'File:C.jpg'], () => left-- > 0)).toEqual({
      'File:A.jpg': false,
      'File:B.jpg': true,
    });
    expect(load.mock.calls.map(([, keys]) => keys)).toEqual([
      ['File:A.jpg', 'File:B.jpg', 'File:C.jpg'],
      ['File:B.jpg', 'File:C.jpg'],
    ]);
  });

  it('stops asking when a truncated answer brings nothing new', async () => {
    const load = vi.fn<LabelFetcher>(async () => ({ labels: new Map(), truncated: true }));
    expect(await new PlatformLookup(load).labels('commons', ['File:A.jpg'], always)).toEqual({});
    expect(load).toHaveBeenCalledTimes(1);
  });

  it('shares one request between overlapping lookups of the same keys', async () => {
    let finish!: () => void;
    const load = vi.fn<LabelFetcher>(
      (_platform, keys) => new Promise((resolve) => (finish = () => resolve({ labels: new Map(keys.map((key) => [key, true])), truncated: false }))),
    );
    const allow = vi.fn(always);
    const lookup = new PlatformLookup(load);
    const first = lookup.labels('pixiv', ['1', '2'], allow);
    const second = lookup.labels('pixiv', ['2', '1'], allow);
    finish();
    expect(await Promise.all([first, second])).toEqual([
      { 1: true, 2: true },
      { 1: true, 2: true },
    ]);
    expect(load).toHaveBeenCalledTimes(1);
    expect(allow).toHaveBeenCalledTimes(1);
  });

  it('ignores keys that are no work ids or file titles and unknown platforms', async () => {
    const load = answering([]);
    const lookup = new PlatformLookup(load);
    await lookup.labels('pixiv', ['1', '2&x=1', 3 as unknown as string], always);
    expect(load).toHaveBeenCalledWith('pixiv', ['1']);
    expect(await lookup.labels('constructor' as 'pixiv', ['1'], always)).toEqual({});
    expect(await lookup.labels('pixiv', 'nope' as unknown as string[], always)).toEqual({});
    expect(load).toHaveBeenCalledTimes(1);
  });
});
