import type { Verdict } from '../shared/messages';
import type { Checks, EncodedImage, WorkerCommand, WorkerRequest, WorkerResponse } from './worker';

export const STALL_TIMEOUT_MS = 60_000;

export interface HeldImage {
  score(): Promise<Verdict>;
  discard(): void;
}

type Answer = Verdict | 'held';

let worker: Worker | null = null;
let nextId = 0;
let stallTimer: ReturnType<typeof setTimeout> | undefined;
let stalled = false;
const pending = new Map<number, (answer: Answer) => void>();
const scoring = new Set<number>();

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
  scoring.clear();
  watchForStall();
}

function ensureWorker(): Worker {
  if (worker) return worker;
  const created = new Worker(browser.runtime.getURL(stalled ? 'worker.js?backend=wasm' : 'worker.js'), { type: 'module' });
  created.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    pending.get(data.id)?.('held' in data ? 'held' : data.verdict);
    pending.delete(data.id);
    if (scoring.delete(data.id) || !scoring.size) watchForStall();
  };
  created.onerror = (event) => discardWorker(`failed: ${event.message}`);
  worker = created;
  return created;
}

function ask(message: WorkerRequest | WorkerCommand, transfer: Transferable[] = []): Promise<Answer> {
  return new Promise((resolve) => {
    pending.set(message.id, resolve);
    const scores = 'command' in message;
    if (scores) scoring.add(message.id);
    if (scores || pending.size === 1) watchForStall();
    ensureWorker().postMessage(message, transfer);
  });
}

export async function prepare(image: EncodedImage, original: EncodedImage | null, checks: Checks): Promise<Verdict | HeldImage> {
  const id = nextId++;
  const answer = await ask({ id, ...image, original, ...checks }, original ? [image.bytes, original.bytes] : [image.bytes]);
  if (answer !== 'held') return answer;
  return {
    score: async () => {
      const verdict = await ask({ id, command: 'score' });
      return verdict === 'held' ? { ok: false, error: 'model' } : verdict;
    },
    discard: () => worker?.postMessage({ id, command: 'drop' } satisfies WorkerCommand),
  };
}
