import type { Verdict } from '../shared/messages';
import type { Checks, EncodedImage, WorkerRequest, WorkerResponse } from './worker';

export const STALL_TIMEOUT_MS = 60_000;

let worker: Worker | null = null;
let nextId = 0;
let stallTimer: ReturnType<typeof setTimeout> | undefined;
let stalled = false;
const pending = new Map<number, (verdict: Verdict) => void>();

function watchForStall(): void {
  clearTimeout(stallTimer);
  stallTimer = pending.size ? setTimeout(onStall, STALL_TIMEOUT_MS) : undefined;
}

function onStall(): void {
  stalled = true;
  discardWorker('gave no answer in time');
}

function discardWorker(reason: string): void {
  console.error('[ai-watermark] worker', reason);
  worker?.terminate();
  worker = null;
  for (const resolve of pending.values()) resolve({ ok: false, error: 'model' });
  pending.clear();
  watchForStall();
}

function ensureWorker(): Worker {
  if (worker) return worker;
  const created = new Worker(browser.runtime.getURL(stalled ? 'worker.js?backend=wasm' : 'worker.js'), { type: 'module' });
  created.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    pending.get(data.id)?.(data.verdict);
    pending.delete(data.id);
    watchForStall();
  };
  created.onerror = (event) => discardWorker(`failed: ${event.message}`);
  worker = created;
  return created;
}

export function evaluate(image: EncodedImage, original: EncodedImage | null, checks: Checks): Promise<Verdict> {
  return new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    if (pending.size === 1) watchForStall();
    const request: WorkerRequest = { id, ...image, original, ...checks };
    ensureWorker().postMessage(request, original ? [image.bytes, original.bytes] : [image.bytes]);
  });
}
