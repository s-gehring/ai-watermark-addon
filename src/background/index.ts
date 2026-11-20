import { LruCache } from '../shared/lru';
import { CANCELLED, type BackgroundRequest, type Verdict } from '../shared/messages';
import { loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { prepare, WORKER_LOST } from './inference';
import { PageBudget } from './page-budget';
import { PlatformLookup } from './platform-lookup';
import { firstResult } from './queue';
import { JobBoard, Lane, type Asker, type Job } from './scheduler';
import { forgetTab, updateTabStats } from './tab-stats';
import type { EncodedImage } from './worker';

const MAX_BYTES = 40 * 1024 * 1024;
const MAX_ORIGINAL_BYTES = 16 * 1024 * 1024;
const MAX_ORIGINAL_BYTES_PER_PAGE = 64 * 1024 * 1024;
const MAX_LABEL_REQUESTS_PER_PAGE = 10;

const cache = new LruCache<string, Verdict>(2000);
const board = new JobBoard<Verdict>();
const inference = new Lane(activeTabs);
const originals = new Lane(activeTabs, 2);
const originalBudget = new PageBudget(MAX_ORIGINAL_BYTES_PER_PAGE);
const labelBudget = new PageBudget(MAX_LABEL_REQUESTS_PER_PAGE);
const paidForOriginal = new WeakMap<Job, number>();
const platformLookup = new PlatformLookup();

let settings: Settings | null = null;
let checksVersion = 0;
const settingsReady = loadSettings().then((s) => (settings = s));
onSettingsChanged((s) => {
  const checksToggled = settings && (settings.checkMetadata !== s.checkMetadata || settings.checkWatermark !== s.checkWatermark);
  settings = s;
  if (!checksToggled) return;
  cache.clear();
  checksVersion++;
});

async function activeTabs(): Promise<ReadonlySet<number>> {
  const tabs = await browser.tabs.query({ active: true, lastFocusedWindow: true }).catch(() => []);
  return new Set(tabs.map((tab) => tab.id ?? -1));
}

async function fetchBytes(url: string, dropped: AbortSignal, maxBytes = MAX_BYTES): Promise<EncodedImage | null> {
  try {
    const signal = AbortSignal.any([dropped, AbortSignal.timeout(20_000)]);
    const res = await fetch(url, { credentials: 'include', cache: 'force-cache', signal });
    if (!res.ok || Number(res.headers.get('content-length')) > maxBytes) {
      void res.body?.cancel();
      return null;
    }
    const buffer = await res.arrayBuffer();
    if (buffer.byteLength === 0 || buffer.byteLength > maxBytes) return null;
    return { bytes: buffer, type: res.headers.get('content-type') ?? '' };
  } catch {
    return null;
  }
}

type ClassifyRequest = Extract<BackgroundRequest, { type: 'classify' }>;

function fetchOriginal(candidates: string[], job: Job): Promise<EncodedImage | null> {
  const load = (maxBytes: number) => firstResult(candidates, (candidate) => fetchBytes(candidate, job.signal, maxBytes));
  return originals.run(job, async () => {
    const paid = paidForOriginal.get(job);
    if (paid) return load(paid);
    const [{ tabId, frameId, page }] = job.requesters;
    const loaded = await originalBudget.load(tabId, frameId, page, () => load(MAX_ORIGINAL_BYTES));
    if (loaded) paidForOriginal.set(job, loaded.bytes.byteLength);
    return loaded;
  });
}

async function attempt({ url, bytes, original }: ClassifyRequest, job: Job): Promise<Verdict | typeof WORKER_LOST> {
  const [fetched, larger] = await Promise.all([
    bytes ? { bytes, type: '' } : fetchBytes(url, job.signal),
    original ? fetchOriginal(original, job) : null,
  ]);
  if (job.dropped) return CANCELLED;
  if (!fetched) return { ok: false, error: 'fetch' };
  if (fetched.type.includes('svg')) return { ok: false, error: 'unsupported' };
  const { checkMetadata, checkWatermark } = settings!;
  const prepared = await prepare(fetched, larger, { checkMetadata, checkWatermark });
  if (prepared === WORKER_LOST || 'ok' in prepared) return prepared;
  const verdict = await inference.run(job, prepared.score);
  if (verdict) return verdict;
  prepared.discard();
  return CANCELLED;
}

async function classify(request: ClassifyRequest, job: Job): Promise<Verdict> {
  await settingsReady;
  let outcome = await attempt(request, job);
  const canLoadAgain = !request.bytes;
  if (outcome === WORKER_LOST && canLoadAgain) outcome = await attempt(request, job);
  return outcome === WORKER_LOST ? { ok: false, error: 'model' } : outcome;
}

function classifyCached(request: ClassifyRequest, asker: Asker): Promise<Verdict> {
  const hit = cache.get(request.url);
  if (hit) return Promise.resolve(hit);
  const version = checksVersion;
  return board.join(`${version} ${request.url}`, asker, async (job) => {
    const verdict = await classify(request, job);
    const final = verdict.ok || verdict.error === 'unsupported' || verdict.error === 'decode';
    if (final && version === checksVersion) cache.set(request.url, verdict);
    return verdict;
  });
}

browser.tabs.onRemoved.addListener((tabId) => {
  board.leaveTab(tabId);
  forgetTab(tabId);
  originalBudget.forgetTab(tabId);
  labelBudget.forgetTab(tabId);
});

browser.runtime.onMessage.addListener((message: unknown, sender) => {
  const request = message as BackgroundRequest;
  const [tabId, frameId] = [sender.tab?.id ?? -1, sender.frameId ?? 0];
  if (request?.type === 'classify') {
    return classifyCached(request, { tabId, frameId, id: request.id, page: request.page });
  }
  if (request?.type === 'out-of-view') {
    board.setOutOfView(tabId, frameId, new Set(request.ids));
    return undefined;
  }
  if (request?.type === 'drop-requests') {
    board.leaveFrame(tabId, frameId);
    return undefined;
  }
  if (request?.type === 'platform-labels') {
    return platformLookup.labels(request.platform, request.keys, () => labelBudget.take(tabId, frameId, request.page));
  }
  if (request?.type === 'page-stats' && sender.tab?.id !== undefined) {
    if (request.initial) {
      if (frameId === 0) board.leaveTab(tabId);
      else board.leaveFrame(tabId, frameId);
      originalBudget.reset(tabId, frameId);
      labelBudget.reset(tabId, frameId);
    }
    void updateTabStats(tabId, frameId, request.stats, request.initial).catch((error) =>
      console.error('[ai-watermark] stats update failed', error),
    );
  }
  return undefined;
});
