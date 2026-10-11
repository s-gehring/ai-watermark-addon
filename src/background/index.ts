import { LruCache } from '../shared/lru';
import type { BackgroundRequest, Verdict } from '../shared/messages';
import { loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { decodeImage, scoreImage } from './classifier';
import { isScorable } from './preprocess';
import { detectAiMetadata } from './metadata';
import { SerialQueue } from './queue';

const MAX_BYTES = 40 * 1024 * 1024;
const BADGE_COLOR = '#6d28d9';

const cache = new LruCache<string, Verdict>(2000);
const inflight = new Map<string, Promise<Verdict>>();
const queue = new SerialQueue();

let settings: Settings | null = null;
const settingsReady = loadSettings().then((s) => (settings = s));
onSettingsChanged((s) => {
  const metadataToggled = settings && settings.checkMetadata !== s.checkMetadata;
  settings = s;
  if (metadataToggled) cache.clear();
});

async function fetchBytes(url: string): Promise<{ bytes: Uint8Array; type: string } | null> {
  try {
    const res = await fetch(url, { credentials: 'include', cache: 'force-cache' });
    if (!res.ok) return null;
    const buffer = await res.arrayBuffer();
    if (buffer.byteLength === 0 || buffer.byteLength > MAX_BYTES) return null;
    return { bytes: new Uint8Array(buffer), type: res.headers.get('content-type') ?? '' };
  } catch {
    return null;
  }
}

async function classify(url: string, provided?: ArrayBuffer): Promise<Verdict> {
  await settingsReady;
  const fetched = provided ? { bytes: new Uint8Array(provided), type: '' } : await fetchBytes(url);
  if (!fetched) return { ok: false, error: 'fetch' };
  if (fetched.type.includes('svg')) return { ok: false, error: 'unsupported' };

  if (settings!.checkMetadata) {
    const reason = detectAiMetadata(fetched.bytes);
    if (reason) return { ok: true, score: 1, source: 'metadata', reason };
  }

  return queue.run(async () => {
    let image;
    try {
      image = await decodeImage(new Blob([fetched.bytes as BlobPart], { type: fetched.type }));
    } catch {
      return { ok: false, error: 'decode' } as const;
    }
    if (!isScorable(image)) return { ok: false, error: 'unsupported' } as const;
    try {
      return { ok: true, score: await scoreImage(image), source: 'model' } as const;
    } catch (error) {
      console.error('[ai-watermark] inference failed', error);
      return { ok: false, error: 'model' } as const;
    }
  });
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

const flaggedPerFrame = new Map<number, Map<number, number>>();

function updateTabBadge(tabId: number, frameId: number, flagged: number, initial: boolean): void {
  if (initial && frameId === 0) flaggedPerFrame.delete(tabId);
  const frames = flaggedPerFrame.get(tabId) ?? new Map<number, number>();
  frames.set(frameId, flagged);
  flaggedPerFrame.set(tabId, frames);
  const total = [...frames.values()].reduce((a, b) => a + b, 0);
  void browser.action.setBadgeBackgroundColor({ tabId, color: BADGE_COLOR });
  void browser.action.setBadgeText({ tabId, text: total > 0 ? String(total) : '' });
}

browser.tabs.onRemoved.addListener((tabId) => flaggedPerFrame.delete(tabId));

browser.runtime.onMessage.addListener((message: unknown, sender) => {
  const request = message as BackgroundRequest;
  if (request?.type === 'classify') {
    return classifyCached(request.url, request.bytes);
  }
  if (request?.type === 'page-stats' && sender.tab?.id !== undefined) {
    updateTabBadge(sender.tab.id, sender.frameId ?? 0, request.flagged, request.initial);
  }
  return undefined;
});
