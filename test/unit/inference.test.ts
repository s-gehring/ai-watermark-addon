import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Verdict } from '../../src/shared/messages';
import type { HeldImage } from '../../src/background/inference';
import type { WorkerCommand, WorkerRequest, WorkerResponse } from '../../src/background/worker';

class FakeWorker {
  static created: FakeWorker[] = [];
  onmessage: ((event: { data: WorkerResponse }) => void) | null = null;
  onerror: ((event: { message: string }) => void) | null = null;
  readonly requests: (WorkerRequest | WorkerCommand)[] = [];
  terminated = false;

  constructor(readonly url: string) {
    FakeWorker.created.push(this);
  }

  postMessage(request: WorkerRequest | WorkerCommand): void {
    this.requests.push(request);
  }

  terminate(): void {
    this.terminated = true;
  }

  answer(index: number, verdict = CLEAN): void {
    this.onmessage!({ data: { id: this.requests[index].id, verdict, backend: 'webgpu' } });
  }

  hold(index: number): void {
    this.onmessage!({ data: { id: this.requests[index].id, held: true } });
  }
}

const CLEAN: Verdict = { ok: true, score: 0.1, models: [0.1], source: 'model', metadata: 'off', watermark: 'off' };
const LABELED: Verdict = { ok: true, score: 1, source: 'metadata', reason: 'comfyui' };
const FAILED: Verdict = { ok: false, error: 'model' };
const CHECKS = { checkMetadata: false, checkWatermark: false };
const image = () => ({ bytes: new ArrayBuffer(4), type: 'image/png' });

let prepare: typeof import('../../src/background/inference').prepare;
let timeout: number;
let keepAwake: number;
let stored: Record<string, unknown>;
const getPlatformInfo = vi.fn(async () => ({}));

async function startBackground(): Promise<void> {
  vi.resetModules();
  FakeWorker.created = [];
  ({ prepare, STALL_TIMEOUT_MS: timeout, KEEP_AWAKE_MS: keepAwake } = await import('../../src/background/inference'));
}

async function request(): Promise<{ prepared: Promise<Verdict | HeldImage> }> {
  const prepared = prepare(image(), null, CHECKS);
  await vi.advanceTimersByTimeAsync(0);
  return { prepared };
}

function settled<T>(promise: Promise<T>): () => T | undefined {
  let result: T | undefined;
  void promise.then((value) => (result = value));
  return () => result;
}

async function held(worker = 0): Promise<HeldImage> {
  const { prepared } = await request();
  const created = FakeWorker.created[worker];
  created.hold(created.requests.length - 1);
  return (await prepared) as HeldImage;
}

beforeEach(async () => {
  vi.useFakeTimers();
  vi.stubGlobal('Worker', FakeWorker);
  stored = {};
  getPlatformInfo.mockClear();
  vi.stubGlobal('browser', {
    runtime: { getURL: (path: string) => path, getPlatformInfo },
    storage: {
      session: {
        get: async (key: string) => (key in stored ? { [key]: stored[key] } : {}),
        set: async (items: Record<string, unknown>) => void Object.assign(stored, items),
      },
    },
  });
  vi.spyOn(console, 'error').mockImplementation(() => {});
  await startBackground();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('prepare', () => {
  it('passes on the verdict when the worker answers at once', async () => {
    const { prepared } = await request();
    FakeWorker.created[0].answer(0, LABELED);
    expect(await prepared).toEqual(LABELED);
  });

  it('scores a held image only when asked and tells the worker to drop one that is discarded', async () => {
    const first = await held();
    const second = await held();
    const [worker] = FakeWorker.created;
    expect(worker.requests).toHaveLength(2);
    const scored = first.score();
    second.discard();
    expect(worker.requests.slice(2)).toEqual([
      { id: worker.requests[0].id, command: 'score' },
      { id: worker.requests[1].id, command: 'drop' },
    ]);
    worker.answer(2);
    expect(await scored).toEqual(CLEAN);
  });

  it('fails every waiting request and replaces the worker when no answer arrives in time', async () => {
    const first = settled((await held()).score());
    const second = settled((await request()).prepared);
    await vi.advanceTimersByTimeAsync(timeout - 1);
    expect(first()).toBeUndefined();
    await vi.advanceTimersByTimeAsync(1);
    expect(first()).toEqual(FAILED);
    expect(second()).toEqual(FAILED);
    expect(FakeWorker.created[0].terminated).toBe(true);

    const retry = settled((await held(1)).score());
    expect(FakeWorker.created.map((worker) => worker.url)).toEqual(['worker.js', 'worker.js?backend=wasm']);
    FakeWorker.created[1].answer(1);
    await vi.advanceTimersByTimeAsync(0);
    expect(retry()).toEqual(CLEAN);
  });

  it('replaces a worker whose model run hangs, however many new images it accepts meanwhile', async () => {
    const hanging = settled((await held()).score());
    await vi.advanceTimersByTimeAsync(timeout - 1);
    await held();
    const { prepared: labeled } = await request();
    FakeWorker.created[0].answer(3, LABELED);
    expect(await labeled).toEqual(LABELED);
    expect(FakeWorker.created[0].terminated).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(hanging()).toEqual(FAILED);
    expect(FakeWorker.created[0].terminated).toBe(true);
  });

  it('gives every model run the full time, also right after another one', async () => {
    const first = settled((await held()).score());
    const next = await held();
    await vi.advanceTimersByTimeAsync(timeout - 1);
    FakeWorker.created[0].answer(1);
    await vi.advanceTimersByTimeAsync(0);
    expect(first()).toEqual(CLEAN);
    const second = settled(next.score());
    await vi.advanceTimersByTimeAsync(timeout - 1);
    expect(second()).toBeUndefined();
    await vi.advanceTimersByTimeAsync(1);
    expect(second()).toEqual(FAILED);
  });

  it('keeps waiting requests alive while the worker keeps answering', async () => {
    const first = settled((await held()).score());
    const second = settled((await request()).prepared);
    const [worker] = FakeWorker.created;
    await vi.advanceTimersByTimeAsync(timeout - 1);
    worker.answer(1);
    await vi.advanceTimersByTimeAsync(timeout - 1);
    expect(first()).toEqual(CLEAN);
    expect(second()).toBeUndefined();
    expect(worker.terminated).toBe(false);
    worker.answer(2);
    await vi.advanceTimersByTimeAsync(0);
    expect(second()).toEqual(CLEAN);
  });

  it('keeps the background page awake with a call into the parent process for as long as a request is open', async () => {
    const waiting = await held();
    await vi.advanceTimersByTimeAsync(3 * keepAwake);
    expect(getPlatformInfo).not.toHaveBeenCalled();
    const scored = settled(waiting.score());
    await vi.advanceTimersByTimeAsync(2 * keepAwake);
    expect(getPlatformInfo).toHaveBeenCalledTimes(2);
    FakeWorker.created[0].answer(1);
    await vi.advanceTimersByTimeAsync(3 * keepAwake);
    expect(scored()).toEqual(CLEAN);
    expect(getPlatformInfo).toHaveBeenCalledTimes(2);
  });

  it('starts on WASM when a model run hung before the background page was suspended', async () => {
    void (await held()).score();
    await vi.advanceTimersByTimeAsync(timeout);
    await startBackground();
    await request();
    expect(FakeWorker.created.map((worker) => worker.url)).toEqual(['worker.js?backend=wasm']);
  });

  it('leaves a worker running that only holds images', async () => {
    await held();
    await vi.advanceTimersByTimeAsync(10 * timeout);
    expect(FakeWorker.created[0].terminated).toBe(false);
  });

  it('fails waiting requests when the worker reports an error', async () => {
    const only = settled((await held()).score());
    FakeWorker.created[0].onerror!({ message: 'out of memory' });
    await vi.advanceTimersByTimeAsync(0);
    expect(only()).toEqual(FAILED);
    expect(FakeWorker.created[0].terminated).toBe(true);
    await request();
    expect(FakeWorker.created[1].url).toBe('worker.js');
  });

  it('starts no worker only to drop an image the failed worker held', async () => {
    const kept = await held();
    FakeWorker.created[0].onerror!({ message: 'out of memory' });
    kept.discard();
    expect(FakeWorker.created).toHaveLength(1);
  });
});
