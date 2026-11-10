import { describe, expect, it, vi } from 'vitest';
import { PlatformLookup, type LabelFetcher } from '../../src/background/platform-lookup';

const always = () => true;
const ids = (count: number, from = 1) => Array.from({ length: count }, (_, i) => String(from + i));

function answering(labeled: string[], complete = true) {
  return vi.fn<LabelFetcher>(async () => ({ labels: new Map(labeled.map((key) => [key, true])), complete }));
}

describe('PlatformLookup', () => {
  it('returns the labeled keys and asks only once per key', async () => {
    const load = answering(['2']);
    const lookup = new PlatformLookup(load);
    expect(await lookup.labeled('pixiv', ['1', '2', '2'], always)).toEqual(['2']);
    expect(await lookup.labeled('pixiv', ['1', '2'], always)).toEqual(['2']);
    expect(load).toHaveBeenCalledTimes(1);
    expect(load).toHaveBeenCalledWith('pixiv', ['1', '2']);
  });

  it('splits many keys into batches', async () => {
    const load = answering([]);
    await new PlatformLookup(load).labeled('pixiv', ids(120), always);
    expect(load.mock.calls.map(([, keys]) => keys.length)).toEqual([50, 50, 20]);
  });

  it('stops asking when the page view has no requests left', async () => {
    const load = answering(['1', '60']);
    let left = 1;
    const lookup = new PlatformLookup(load);
    expect(await lookup.labeled('pixiv', ids(120), () => left-- > 0)).toEqual(['1']);
    expect(load).toHaveBeenCalledTimes(1);
    expect(await lookup.labeled('pixiv', ['60'], always)).toEqual(['60']);
  });

  it('asks again after a failed request', async () => {
    const load = vi.fn<LabelFetcher>(async () => null);
    const lookup = new PlatformLookup(load);
    expect(await lookup.labeled('commons', ['File:A.jpg'], always)).toEqual([]);
    expect(await lookup.labeled('commons', ['File:A.jpg'], always)).toEqual([]);
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('does not remember missing labels from a truncated answer', async () => {
    const load = answering(['File:A.jpg'], false);
    const lookup = new PlatformLookup(load);
    expect(await lookup.labeled('commons', ['File:A.jpg', 'File:B.jpg'], always)).toEqual(['File:A.jpg']);
    await lookup.labeled('commons', ['File:A.jpg', 'File:B.jpg'], always);
    expect(load.mock.calls.map(([, keys]) => keys)).toEqual([['File:A.jpg', 'File:B.jpg'], ['File:B.jpg']]);
  });

  it('ignores keys that are no work ids or file titles and unknown platforms', async () => {
    const load = answering([]);
    const lookup = new PlatformLookup(load);
    await lookup.labeled('pixiv', ['1', '2&x=1', 3 as unknown as string], always);
    expect(load).toHaveBeenCalledWith('pixiv', ['1']);
    expect(await lookup.labeled('constructor' as 'pixiv', ['1'], always)).toEqual([]);
    expect(await lookup.labeled('pixiv', 'nope' as unknown as string[], always)).toEqual([]);
    expect(load).toHaveBeenCalledTimes(1);
  });
});
