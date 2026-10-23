import { LruCache } from '../shared/lru';
import type { BackgroundRequest, Verdict } from '../shared/messages';
import { loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { evaluate } from './inference';
import { firstResult, TaskQueue } from './queue';
import { forgetTab, updateTabStats } from './tab-stats';
import type { EncodedImage } from './worker';

const MAX_BYTES = 40 * 1024 * 1024;
const MAX_ORIGINAL_BYTES = 16 * 1024 * 1024;

const cache = new LruCache<string, Verdict>(2000);
const inflight = new Map<string, Promise<Verdict>>();
const originals = new TaskQueue(2);

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

async function classify(url: string, provided?: ArrayBuffer, original?: string[]): Promise<Verdict> {
  await settingsReady;
  const [fetched, larger] = await Promise.all([
    provided ? { bytes: provided, type: '' } : fetchBytes(url),
    original ? originals.run(() => firstResult(original, (candidate) => fetchBytes(candidate, MAX_ORIGINAL_BYTES))) : null,
  ]);
  if (!fetched) return { ok: false, error: 'fetch' };
  if (fetched.type.includes('svg')) return { ok: false, error: 'unsupported' };
  const { checkMetadata, checkWatermark } = settings!;
  return evaluate(fetched, larger, { checkMetadata, checkWatermark });
}

function classifyCached(url: string, bytes?: ArrayBuffer, original?: string[]): Promise<Verdict> {
  const hit = cache.get(url);
  if (hit) return Promise.resolve(hit);
  let pending = inflight.get(url);
  if (!pending) {
    pending = classify(url, bytes, original)
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
    return classifyCached(request.url, request.bytes, request.original);
  }
  if (request?.type === 'page-stats' && sender.tab?.id !== undefined) {
    void updateTabStats(sender.tab.id, sender.frameId ?? 0, request.stats, request.initial).catch((error) =>
      console.error('[ai-watermark] stats update failed', error),
    );
  }
  return undefined;
});
