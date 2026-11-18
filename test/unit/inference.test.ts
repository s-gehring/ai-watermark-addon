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

function settled<T>(promise: Promise<T>): () => T | undefined {
  let result: T | undefined;
  void promise.then((value) => (result = value));
  return () => result;
}

async function held(worker = 0): Promise<HeldImage> {
  const prepared = prepare(image(), null, CHECKS);
  const created = FakeWorker.created[worker];
  created.hold(created.requests.length - 1);
  return (await prepared) as HeldImage;
}

beforeEach(async () => {
  vi.resetModules();
  vi.useFakeTimers();
  vi.stubGlobal('Worker', FakeWorker);
  vi.stubGlobal('browser', { runtime: { getURL: (path: string) => path } });
  vi.spyOn(console, 'error').mockImplementation(() => {});
  FakeWorker.created = [];
  ({ prepare, STALL_TIMEOUT_MS: timeout } = await import('../../src/background/inference'));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('prepare', () => {
  it('passes on the verdict when the worker answers at once', async () => {
    const prepared = prepare(image(), null, CHECKS);
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
    const second = settled(prepare(image(), null, CHECKS));
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

  it('keeps waiting requests alive while the worker keeps answering', async () => {
    const first = settled((await held()).score());
    const second = settled(prepare(image(), null, CHECKS));
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
    void prepare(image(), null, CHECKS);
    expect(FakeWorker.created[1].url).toBe('worker.js');
  });

  it('starts no worker only to drop an image the failed worker held', async () => {
    const kept = await held();
    FakeWorker.created[0].onerror!({ message: 'out of memory' });
    kept.discard();
    expect(FakeWorker.created).toHaveLength(1);
  });
});
