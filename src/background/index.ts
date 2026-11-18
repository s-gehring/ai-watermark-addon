import { LruCache } from '../shared/lru';
import { CANCELLED, type BackgroundRequest, type Verdict } from '../shared/messages';
import { loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { prepare } from './inference';
import { PageBudget } from './page-budget';
import { PlatformLookup } from './platform-lookup';
import { firstResult } from './queue';
import { JobBoard, Lane, type Job, type Requester } from './scheduler';
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

interface Request {
  url: string;
  page: string;
  bytes?: ArrayBuffer;
  original?: string[];
}

function fetchOriginal(candidates: string[], page: string, { tabId, frameId }: Requester, job: Job): Promise<EncodedImage | null> {
  const load = () => firstResult(candidates, (candidate) => fetchBytes(candidate, MAX_ORIGINAL_BYTES));
  return originals.run(job, () => originalBudget.load(tabId, frameId, page, load));
}

async function classify({ url, page, bytes, original }: Request, requester: Requester, job: Job): Promise<Verdict> {
  await settingsReady;
  const [fetched, larger] = await Promise.all([
    bytes ? { bytes, type: '' } : fetchBytes(url),
    original ? fetchOriginal(original, page, requester, job) : null,
  ]);
  if (job.dropped) return CANCELLED;
  if (!fetched) return { ok: false, error: 'fetch' };
  if (fetched.type.includes('svg')) return { ok: false, error: 'unsupported' };
  const { checkMetadata, checkWatermark } = settings!;
  const held = await prepare(fetched, larger, { checkMetadata, checkWatermark });
  if ('ok' in held) return held;
  const verdict = await inference.run(job, held.score);
  if (verdict) return verdict;
  held.discard();
  return CANCELLED;
}

function classifyCached(request: Request, requester: Requester): Promise<Verdict> {
  const hit = cache.get(request.url);
  if (hit) return Promise.resolve(hit);
  return board.join(request.url, requester, async (job) => {
    const verdict = await classify(request, requester, job);
    if (verdict.ok || verdict.error === 'unsupported' || verdict.error === 'decode') cache.set(request.url, verdict);
    return verdict;
  });
}

browser.tabs.onRemoved.addListener((tabId) => {
  board.leave((requester) => requester.tabId === tabId);
  forgetTab(tabId);
  originalBudget.forgetTab(tabId);
  labelBudget.forgetTab(tabId);
});

browser.runtime.onMessage.addListener((message: unknown, sender) => {
  const request = message as BackgroundRequest;
  const [tabId, frameId] = [sender.tab?.id ?? -1, sender.frameId ?? 0];
  const sameFrame = (requester: Requester) => requester.tabId === tabId && requester.frameId === frameId;
  if (request?.type === 'classify') {
    return classifyCached(request, { tabId, frameId, id: request.id, inView: true });
  }
  if (request?.type === 'out-of-view') {
    board.setOutOfView(tabId, frameId, new Set(request.ids));
    return undefined;
  }
  if (request?.type === 'page-gone') {
    board.leave(sameFrame);
    return undefined;
  }
  if (request?.type === 'platform-labels') {
    return platformLookup.labels(request.platform, request.keys, () => labelBudget.take(tabId, frameId, request.page));
  }
  if (request?.type === 'page-stats' && sender.tab?.id !== undefined) {
    if (request.initial) {
      board.leave(sameFrame);
      originalBudget.reset(tabId, frameId);
      labelBudget.reset(tabId, frameId);
    }
    void updateTabStats(tabId, frameId, request.stats, request.initial).catch((error) =>
      console.error('[ai-watermark] stats update failed', error),
    );
  }
  return undefined;
});
