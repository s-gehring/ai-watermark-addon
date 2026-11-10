import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Verdict } from '../../src/shared/messages';
import type { WorkerRequest, WorkerResponse } from '../../src/background/worker';

class FakeWorker {
  static created: FakeWorker[] = [];
  onmessage: ((event: { data: WorkerResponse }) => void) | null = null;
  onerror: ((event: { message: string }) => void) | null = null;
  readonly requests: WorkerRequest[] = [];
  terminated = false;

  constructor() {
    FakeWorker.created.push(this);
  }

  postMessage(request: WorkerRequest): void {
    this.requests.push(request);
  }

  terminate(): void {
    this.terminated = true;
  }

  answer(index: number): void {
    this.onmessage!({ data: { id: this.requests[index].id, verdict: CLEAN } });
  }
}

const CLEAN: Verdict = { ok: true, score: 0.1, models: [0.1], source: 'model', metadata: 'off', watermark: 'off' };
const FAILED: Verdict = { ok: false, error: 'model' };
const CHECKS = { checkMetadata: false, checkWatermark: false };
const image = () => ({ bytes: new ArrayBuffer(4), type: 'image/png' });

let evaluate: typeof import('../../src/background/inference').evaluate;
let timeout: number;

function settled(promise: Promise<Verdict>): () => Verdict | undefined {
  let result: Verdict | undefined;
  void promise.then((verdict) => (result = verdict));
  return () => result;
}

beforeEach(async () => {
  vi.resetModules();
  vi.useFakeTimers();
  vi.stubGlobal('Worker', FakeWorker);
  vi.stubGlobal('browser', { runtime: { getURL: (path: string) => path } });
  vi.spyOn(console, 'error').mockImplementation(() => {});
  FakeWorker.created = [];
  ({ evaluate, STALL_TIMEOUT_MS: timeout } = await import('../../src/background/inference'));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('evaluate', () => {
  it('fails every waiting request and replaces the worker when no answer arrives in time', async () => {
    const first = settled(evaluate(image(), null, CHECKS));
    const second = settled(evaluate(image(), null, CHECKS));
    await vi.advanceTimersByTimeAsync(timeout - 1);
    expect(first()).toBeUndefined();
    await vi.advanceTimersByTimeAsync(1);
    expect(first()).toEqual(FAILED);
    expect(second()).toEqual(FAILED);
    expect(FakeWorker.created[0].terminated).toBe(true);

    const retry = settled(evaluate(image(), null, CHECKS));
    expect(FakeWorker.created).toHaveLength(2);
    FakeWorker.created[1].answer(0);
    await vi.advanceTimersByTimeAsync(0);
    expect(retry()).toEqual(CLEAN);
  });

  it('keeps waiting requests alive while the worker keeps answering', async () => {
    const first = settled(evaluate(image(), null, CHECKS));
    const second = settled(evaluate(image(), null, CHECKS));
    const [worker] = FakeWorker.created;
    await vi.advanceTimersByTimeAsync(timeout - 1);
    worker.answer(0);
    await vi.advanceTimersByTimeAsync(timeout - 1);
    expect(first()).toEqual(CLEAN);
    expect(second()).toBeUndefined();
    expect(worker.terminated).toBe(false);
    worker.answer(1);
    await vi.advanceTimersByTimeAsync(0);
    expect(second()).toEqual(CLEAN);
  });

  it('leaves an idle worker running', async () => {
    const only = settled(evaluate(image(), null, CHECKS));
    FakeWorker.created[0].answer(0);
    await vi.advanceTimersByTimeAsync(10 * timeout);
    expect(only()).toEqual(CLEAN);
    expect(FakeWorker.created[0].terminated).toBe(false);
  });

  it('fails waiting requests when the worker reports an error', async () => {
    const only = settled(evaluate(image(), null, CHECKS));
    FakeWorker.created[0].onerror!({ message: 'out of memory' });
    await vi.advanceTimersByTimeAsync(0);
    expect(only()).toEqual(FAILED);
    expect(FakeWorker.created[0].terminated).toBe(true);
  });
});
