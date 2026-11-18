import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { BackgroundRequest, Verdict } from '../../src/shared/messages';

const prepare = vi.hoisted(() => vi.fn());

vi.mock('../../src/background/inference', () => ({ prepare }));
vi.mock('../../src/background/tab-stats', () => ({ forgetTab: vi.fn(), updateTabStats: vi.fn(async () => {}) }));
vi.mock('../../src/background/platform-lookup', () => ({ PlatformLookup: class {} }));
vi.mock('../../src/shared/settings', () => ({
  loadSettings: async () => ({ checkMetadata: true, checkWatermark: true }),
  onSettingsChanged: () => {},
}));

const CLEAN: Verdict = { ok: true, score: 0.1, models: [0.1], source: 'model', metadata: 'clear', watermark: 'clear' };
const LABELED: Verdict = { ok: true, score: 1, source: 'metadata', reason: 'comfyui' };
const CANCELLED: Verdict = { ok: false, error: 'cancelled' };
const STATS = { flagged: 0, possible: 0, clean: 0, pending: 0, failed: 0 };

type Sender = { tab: { id: number }; frameId: number };

let onMessage: (message: BackgroundRequest, sender: Sender) => Promise<Verdict> | undefined;
let onTabRemoved: (tabId: number) => void;
let fetched: ReturnType<typeof vi.fn>;
let nextId = 0;

const from = (tabId: number, frameId = 0): Sender => ({ tab: { id: tabId }, frameId });
const classify = (url: string, sender: Sender) => onMessage({ type: 'classify', id: nextId++, url, page: 'https://page.test/' }, sender)!;

function heldImage(): { image: { score: ReturnType<typeof vi.fn>; discard: ReturnType<typeof vi.fn> }; finish: () => void } {
  let finish = () => {};
  const scored = new Promise<Verdict>((resolve) => (finish = () => resolve(CLEAN)));
  return { image: { score: vi.fn(() => scored), discard: vi.fn() }, finish };
}

async function running(url: string, sender: Sender): Promise<{ answer: Promise<Verdict>; finish: () => void }> {
  const { image, finish } = heldImage();
  prepare.mockResolvedValueOnce(image);
  const answer = classify(url, sender);
  await vi.waitFor(() => expect(image.score).toHaveBeenCalled());
  return { answer, finish };
}

async function waiting(url: string, sender: Sender): Promise<{ answer: Promise<Verdict>; image: ReturnType<typeof heldImage>['image'] }> {
  const { image } = heldImage();
  const calls = prepare.mock.calls.length;
  prepare.mockResolvedValueOnce(image);
  const answer = classify(url, sender);
  await vi.waitFor(() => expect(prepare).toHaveBeenCalledTimes(calls + 1));
  return { answer, image };
}

beforeEach(async () => {
  vi.resetModules();
  prepare.mockReset();
  fetched = vi.fn(async () => new Response(new Uint8Array([1, 2, 3]), { headers: { 'content-type': 'image/jpeg' } }));
  vi.stubGlobal('fetch', fetched);
  vi.stubGlobal('browser', {
    tabs: {
      query: async () => [{ id: 1 }],
      onRemoved: { addListener: (listener: typeof onTabRemoved) => (onTabRemoved = listener) },
    },
    runtime: { onMessage: { addListener: (listener: typeof onMessage) => (onMessage = listener) } },
  });
  await import('../../src/background/index');
});

afterEach(() => vi.unstubAllGlobals());

describe('background', () => {
  it('answers the waiting requests of a frame with cancelled when that frame starts a new page', async () => {
    const first = await running('https://a.test/1.jpg', from(1));
    const second = await waiting('https://a.test/2.jpg', from(1));
    const inFrame = await waiting('https://a.test/3.jpg', from(1, 5));
    void onMessage({ type: 'page-stats', stats: STATS, initial: true }, from(1));
    first.finish();
    expect(await first.answer).toEqual(CLEAN);
    expect(await second.answer).toEqual(CANCELLED);
    expect(second.image.score).not.toHaveBeenCalled();
    expect(second.image.discard).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(inFrame.image.score).toHaveBeenCalled());
  });

  it('drops the waiting requests of a frame that asks for it and of a tab that is closed', async () => {
    const first = await running('https://a.test/1.jpg', from(1));
    const dropped = await waiting('https://a.test/2.jpg', from(1, 5));
    const closed = await waiting('https://a.test/3.jpg', from(2));
    const kept = await waiting('https://a.test/4.jpg', from(3));
    void onMessage({ type: 'drop-requests' }, from(1, 5));
    onTabRemoved(2);
    first.finish();
    expect(await dropped.answer).toEqual(CANCELLED);
    expect(await closed.answer).toEqual(CANCELLED);
    await vi.waitFor(() => expect(kept.image.score).toHaveBeenCalled());
  });

  it('keeps a request that another tab still waits for', async () => {
    const first = await running('https://a.test/1.jpg', from(1));
    const shared = await waiting('https://a.test/2.jpg', from(1));
    const other = classify('https://a.test/2.jpg', from(2));
    onTabRemoved(1);
    first.finish();
    await vi.waitFor(() => expect(shared.image.score).toHaveBeenCalled());
    expect(shared.answer).toBe(other);
  });

  it('does not hand an image to the worker when its page left during the download, and asks again later', async () => {
    let arrive = () => {};
    const download = new Promise<Response>((resolve) => {
      arrive = () => resolve(new Response(new Uint8Array([1])));
    });
    fetched.mockReturnValueOnce(download);
    const left = classify('https://a.test/1.jpg', from(1));
    await vi.waitFor(() => expect(fetched).toHaveBeenCalledTimes(1));
    void onMessage({ type: 'drop-requests' }, from(1));
    arrive();
    expect(await left).toEqual(CANCELLED);
    expect(prepare).not.toHaveBeenCalled();

    prepare.mockResolvedValueOnce(LABELED);
    expect(await classify('https://a.test/1.jpg', from(1))).toEqual(LABELED);
    expect(fetched).toHaveBeenCalledTimes(2);
    expect(await classify('https://a.test/1.jpg', from(2))).toEqual(LABELED);
    expect(fetched).toHaveBeenCalledTimes(2);
  });

  it('checks what a frame reported as out of view after the images it still shows', async () => {
    const first = await running('https://a.test/1.jpg', from(1));
    const scrolledPast = await waiting('https://a.test/2.jpg', from(1));
    const inView = await waiting('https://a.test/3.jpg', from(1));
    void onMessage({ type: 'out-of-view', ids: [nextId - 2] }, from(1));
    first.finish();
    await vi.waitFor(() => expect(inView.image.score).toHaveBeenCalled());
    expect(scrolledPast.image.score).not.toHaveBeenCalled();
  });
});
