import type { Verdict } from '../shared/messages';
import type { Checks, EncodedImage, WorkerCommand, WorkerRequest, WorkerResponse } from './worker';

export const STALL_TIMEOUT_MS = 60_000;
export const KEEP_AWAKE_MS = 10_000;
const STALLED_KEY = 'modelStalled';

export const WORKER_LOST = Symbol('worker lost');
type Lost = typeof WORKER_LOST;

export interface HeldImage {
  score(): Promise<Verdict | Lost>;
  discard(): void;
}

type Answer = Verdict | 'held' | Lost;
type Backend = WorkerResponse['backend'];

const FAILED: Verdict = { ok: false, error: 'model' };

let worker: Worker | null = null;
let nextId = 0;
let stallTimer: ReturnType<typeof setTimeout> | undefined;
let keepAwake: ReturnType<typeof setInterval> | undefined;
let stalled = false;
let backend: Backend | null = null;
interface Open {
  resolve(answer: Answer): void;
  reported: Backend | null;
}

const pending = new Map<number, Open>();
const scoring = new Set<number>();
const lost = new WeakSet<Worker>();
const stalledBefore = browser.storage.session.get(STALLED_KEY).then(
  (stored) => {
    stalled ||= stored[STALLED_KEY] === true;
  },
  () => {},
);

// Only a call into the parent process resets the idle timer that suspends this event page; timers and open requests do not.
function stayAwake(): void {
  void browser.runtime.getPlatformInfo().catch(() => {});
}

function watchForStall(): void {
  clearTimeout(stallTimer);
  clearInterval(keepAwake);
  if (!pending.size) return;
  stallTimer = setTimeout(onStall, STALL_TIMEOUT_MS);
  keepAwake = setInterval(stayAwake, KEEP_AWAKE_MS);
}

function settle(id: number, answer: Answer): void {
  pending.get(id)?.resolve(answer);
  pending.delete(id);
}

function onStall(): void {
  stalled = true;
  void browser.storage.session.set({ [STALLED_KEY]: true }).catch(() => {});
  const [id, hanging] = pending.entries().next().value ?? [];
  if (id !== undefined && hanging?.reported !== 'webgpu') settle(id, FAILED);
  discardWorker('gave no answer in time');
}

function discardWorker(reason: string): void {
  console.error('[ai-watermark] worker', reason);
  if (worker) lost.add(worker);
  worker?.terminate();
  worker = null;
  backend = null;
  for (const { resolve } of pending.values()) resolve(WORKER_LOST);
  pending.clear();
  scoring.clear();
  watchForStall();
}

function ensureWorker(): Worker {
  if (worker) return worker;
  const created = new Worker(browser.runtime.getURL(stalled ? 'worker.js?backend=wasm' : 'worker.js'), { type: 'module' });
  created.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    backend = data.backend;
    settle(data.id, 'held' in data ? 'held' : data.verdict);
    if (scoring.delete(data.id) || !scoring.size) watchForStall();
  };
  created.onerror = (event) => discardWorker(`failed: ${event.message}`);
  worker = created;
  return created;
}

function ask(message: WorkerRequest | WorkerCommand, transfer: Transferable[] = []): Promise<Answer> {
  return new Promise((resolve) => {
    pending.set(message.id, { resolve, reported: backend });
    const scores = 'command' in message;
    if (scores) scoring.add(message.id);
    if (scores || pending.size === 1) watchForStall();
    ensureWorker().postMessage(message, transfer);
  });
}

export async function prepare(image: EncodedImage, original: EncodedImage | null, checks: Checks): Promise<Verdict | HeldImage | Lost> {
  await stalledBefore;
  const id = nextId++;
  const holder = ensureWorker();
  const answer = await ask({ id, ...image, original, ...checks }, original ? [image.bytes, original.bytes] : [image.bytes]);
  if (answer !== 'held') return answer;
  return {
    score: async () => {
      if (lost.has(holder)) return WORKER_LOST;
      const verdict = await ask({ id, command: 'score' });
      return verdict === 'held' ? FAILED : verdict;
    },
    discard: () => worker?.postMessage({ id, command: 'drop' } satisfies WorkerCommand),
  };
}
