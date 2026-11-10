import { LruCache } from '../shared/lru';
import type { BackgroundRequest, Verdict } from '../shared/messages';
import { loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { evaluate } from './inference';
import { PageBudget } from './page-budget';
import { PlatformLookup } from './platform-lookup';
import { firstResult, TaskQueue } from './queue';
import { forgetTab, updateTabStats } from './tab-stats';
import type { EncodedImage } from './worker';

const MAX_BYTES = 40 * 1024 * 1024;
const MAX_ORIGINAL_BYTES = 16 * 1024 * 1024;
const MAX_ORIGINAL_BYTES_PER_PAGE = 64 * 1024 * 1024;
const MAX_LABEL_REQUESTS_PER_PAGE = 10;

const cache = new LruCache<string, Verdict>(2000);
const inflight = new Map<string, Promise<Verdict>>();
const originals = new TaskQueue(2);
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

interface PageView {
  tabId: number;
  frameId: number;
  page: string;
}

function fetchOriginal(candidates: string[], { tabId, frameId, page }: PageView): Promise<EncodedImage | null> {
  const load = () => firstResult(candidates, (candidate) => fetchBytes(candidate, MAX_ORIGINAL_BYTES));
  return originals.run(() => originalBudget.load(tabId, frameId, page, load));
}

async function classify(url: string, view: PageView, provided?: ArrayBuffer, original?: string[]): Promise<Verdict> {
  await settingsReady;
  const [fetched, larger] = await Promise.all([
    provided ? { bytes: provided, type: '' } : fetchBytes(url),
    original ? fetchOriginal(original, view) : null,
  ]);
  if (!fetched) return { ok: false, error: 'fetch' };
  if (fetched.type.includes('svg')) return { ok: false, error: 'unsupported' };
  const { checkMetadata, checkWatermark } = settings!;
  return evaluate(fetched, larger, { checkMetadata, checkWatermark });
}

function classifyCached(url: string, view: PageView, bytes?: ArrayBuffer, original?: string[]): Promise<Verdict> {
  const hit = cache.get(url);
  if (hit) return Promise.resolve(hit);
  let pending = inflight.get(url);
  if (!pending) {
    pending = classify(url, view, bytes, original)
      .then((verdict) => {
        if (verdict.ok || verdict.error === 'unsupported' || verdict.error === 'decode') cache.set(url, verdict);
        return verdict;
      })
      .finally(() => inflight.delete(url));
    inflight.set(url, pending);
  }
  return pending;
}

browser.tabs.onRemoved.addListener((tabId) => {
  forgetTab(tabId);
  originalBudget.forgetTab(tabId);
  labelBudget.forgetTab(tabId);
});

browser.runtime.onMessage.addListener((message: unknown, sender) => {
  const request = message as BackgroundRequest;
  if (request?.type === 'classify') {
    const view = { tabId: sender.tab?.id ?? -1, frameId: sender.frameId ?? 0, page: request.page };
    return classifyCached(request.url, view, request.bytes, request.original);
  }
  if (request?.type === 'platform-labels') {
    const [tabId, frameId] = [sender.tab?.id ?? -1, sender.frameId ?? 0];
    return platformLookup.labels(request.platform, request.keys, () => labelBudget.take(tabId, frameId, request.page));
  }
  if (request?.type === 'page-stats' && sender.tab?.id !== undefined) {
    if (request.initial) {
      originalBudget.reset(sender.tab.id, sender.frameId ?? 0);
      labelBudget.reset(sender.tab.id, sender.frameId ?? 0);
    }
    void updateTabStats(sender.tab.id, sender.frameId ?? 0, request.stats, request.initial).catch((error) =>
      console.error('[ai-watermark] stats update failed', error),
    );
  }
  return undefined;
});
