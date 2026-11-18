import { LruCache } from '../shared/lru';
import { CANCELLED, type BackgroundRequest, type Verdict } from '../shared/messages';
import { loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { prepare } from './inference';
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
const platformLookup = new PlatformLookup();

let settings: Settings | null = null;
const settingsReady = loadSettings().then((s) => (settings = s));
onSettingsChanged((s) => {
  const checksToggled = settings && (settings.checkMetadata !== s.checkMetadata || settings.checkWatermark !== s.checkWatermark);
  settings = s;
  if (checksToggled) cache.clear();
});

async function activeTabs(): Promise<ReadonlySet<number>> {
  const tabs = await browser.tabs.query({ active: true }).catch(() => []);
  return new Set(tabs.map((tab) => tab.id ?? -1));
}

async function fetchBytes(url: string, maxBytes = MAX_BYTES): Promise<EncodedImage | null> {
  try {
    const res = await fetch(url, { credentials: 'include', cache: 'force-cache', signal: AbortSignal.timeout(20_000) });
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
  const load = () => firstResult(candidates, (candidate) => fetchBytes(candidate, MAX_ORIGINAL_BYTES));
  return originals.run(job, () => {
    const [{ tabId, frameId, page }] = job.requesters;
    return originalBudget.load(tabId, frameId, page, load);
  });
}

async function classify({ url, bytes, original }: ClassifyRequest, job: Job): Promise<Verdict> {
  await settingsReady;
  const [fetched, larger] = await Promise.all([bytes ? { bytes, type: '' } : fetchBytes(url), original ? fetchOriginal(original, job) : null]);
  if (job.dropped) return CANCELLED;
  if (!fetched) return { ok: false, error: 'fetch' };
  if (fetched.type.includes('svg')) return { ok: false, error: 'unsupported' };
  const { checkMetadata, checkWatermark } = settings!;
  const prepared = await prepare(fetched, larger, { checkMetadata, checkWatermark });
  if ('ok' in prepared) return prepared;
  const verdict = await inference.run(job, prepared.score);
  if (verdict) return verdict;
  prepared.discard();
  return CANCELLED;
}

function classifyCached(request: ClassifyRequest, asker: Asker): Promise<Verdict> {
  const hit = cache.get(request.url);
  if (hit) return Promise.resolve(hit);
  return board.join(request.url, asker, async (job) => {
    const verdict = await classify(request, job);
    if (verdict.ok || verdict.error === 'unsupported' || verdict.error === 'decode') cache.set(request.url, verdict);
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
      board.leaveFrame(tabId, frameId);
      originalBudget.reset(tabId, frameId);
      labelBudget.reset(tabId, frameId);
    }
    void updateTabStats(tabId, frameId, request.stats, request.initial).catch((error) =>
      console.error('[ai-watermark] stats update failed', error),
    );
  }
  return undefined;
});
