import { LruCache } from '../shared/lru';
import type { BackgroundRequest, Verdict } from '../shared/messages';
import { loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { evaluate } from './inference';
import { forgetTab, updateTabStats } from './tab-stats';

const MAX_BYTES = 40 * 1024 * 1024;

const cache = new LruCache<string, Verdict>(2000);
const inflight = new Map<string, Promise<Verdict>>();

let settings: Settings | null = null;
const settingsReady = loadSettings().then((s) => (settings = s));
onSettingsChanged((s) => {
  const metadataToggled = settings && settings.checkMetadata !== s.checkMetadata;
  settings = s;
  if (metadataToggled) cache.clear();
});

async function fetchBytes(url: string): Promise<{ bytes: ArrayBuffer; type: string } | null> {
  try {
    const res = await fetch(url, { credentials: 'include', cache: 'force-cache', signal: AbortSignal.timeout(20_000) });
    if (!res.ok) return null;
    const buffer = await res.arrayBuffer();
    if (buffer.byteLength === 0 || buffer.byteLength > MAX_BYTES) return null;
    return { bytes: buffer, type: res.headers.get('content-type') ?? '' };
  } catch {
    return null;
  }
}

async function classify(url: string, provided?: ArrayBuffer): Promise<Verdict> {
  await settingsReady;
  const fetched = provided ? { bytes: provided, type: '' } : await fetchBytes(url);
  if (!fetched) return { ok: false, error: 'fetch' };
  if (fetched.type.includes('svg')) return { ok: false, error: 'unsupported' };
  return evaluate(fetched.bytes, fetched.type, settings!.checkMetadata);
}

function classifyCached(url: string, bytes?: ArrayBuffer): Promise<Verdict> {
  const hit = cache.get(url);
  if (hit) return Promise.resolve(hit);
  let pending = inflight.get(url);
  if (!pending) {
    pending = classify(url, bytes)
      .then((verdict) => {
        if (verdict.ok || verdict.error === 'unsupported' || verdict.error === 'decode') cache.set(url, verdict);
        return verdict;
      })
      .finally(() => inflight.delete(url));
    inflight.set(url, pending);
  }
  return pending;
}

browser.tabs.onRemoved.addListener(forgetTab);

browser.runtime.onMessage.addListener((message: unknown, sender) => {
  const request = message as BackgroundRequest;
  if (request?.type === 'classify') {
    return classifyCached(request.url, request.bytes);
  }
  if (request?.type === 'page-stats' && sender.tab?.id !== undefined) {
    void updateTabStats(sender.tab.id, sender.frameId ?? 0, request.stats, request.initial).catch((error) =>
      console.error('[ai-watermark] stats update failed', error),
    );
  }
  return undefined;
});
