import type { Verdict } from '../shared/messages';
import type { WorkerRequest, WorkerResponse } from './worker';

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<number, (verdict: Verdict) => void>();

function failAll(): void {
  for (const resolve of pending.values()) resolve({ ok: false, error: 'model' });
  pending.clear();
}

function ensureWorker(): Worker {
  if (worker) return worker;
  const created = new Worker(browser.runtime.getURL('worker.js'), { type: 'module' });
  created.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    pending.get(data.id)?.(data.verdict);
    pending.delete(data.id);
  };
  created.onerror = (event) => {
    console.error('[ai-watermark] worker failed', event.message);
    created.terminate();
    worker = null;
    failAll();
  };
  worker = created;
  return created;
}

export function evaluate(bytes: ArrayBuffer, type: string, checkMetadata: boolean): Promise<Verdict> {
  return new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    const request: WorkerRequest = { id, bytes, type, checkMetadata };
    ensureWorker().postMessage(request, [bytes]);
  });
}
