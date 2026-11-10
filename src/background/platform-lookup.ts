import { LruCache } from '../shared/lru';
import { isLookupKey, LOOKUP_BATCH, lookupBatches, lookupUrl, parseLookup, type LookupPlatform, type LookupResult } from '../shared/platform-labels';

const MAX_KEYS_PER_MESSAGE = 500;
const TIMEOUT_MS = 5_000;

export type LabelFetcher = (platform: LookupPlatform, keys: string[]) => Promise<LookupResult | null>;

async function fetchLabels(platform: LookupPlatform, keys: string[]): Promise<LookupResult | null> {
  try {
    const res = await fetch(lookupUrl(platform, keys), {
      credentials: platform === 'pixiv' ? 'include' : 'omit',
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (!res.ok) {
      void res.body?.cancel();
      return null;
    }
    return parseLookup(platform, await res.json());
  } catch {
    return null;
  }
}

export class PlatformLookup {
  private readonly cache = new LruCache<string, boolean>(5000);
  private readonly inflight = new Map<string, Promise<void>>();

  constructor(private readonly load: LabelFetcher = fetchLabels) {}

  async labels(platform: LookupPlatform, keys: string[], allowRequest: () => boolean): Promise<Record<string, boolean>> {
    if (!Object.hasOwn(LOOKUP_BATCH, platform) || !Array.isArray(keys)) return {};
    const id = (key: string) => `${platform}:${key}`;
    const wanted = [...new Set(keys)].filter((key) => typeof key === 'string' && isLookupKey(platform, key)).slice(0, MAX_KEYS_PER_MESSAGE);
    const open = wanted.filter((key) => this.cache.get(id(key)) === undefined && !this.inflight.has(id(key)));
    if (open.length) {
      const asked = this.ask(platform, open, allowRequest).finally(() => open.forEach((key) => this.inflight.delete(id(key))));
      for (const key of open) this.inflight.set(id(key), asked);
    }
    await Promise.all(wanted.map((key) => this.inflight.get(id(key))));
    const answers: Record<string, boolean> = {};
    for (const key of wanted) {
      const label = this.cache.get(id(key));
      if (label !== undefined) answers[key] = label;
    }
    return answers;
  }

  private async ask(platform: LookupPlatform, keys: string[], allowRequest: () => boolean): Promise<void> {
    for (const batch of lookupBatches(platform, keys)) {
      let open = batch;
      while (open.length && allowRequest()) {
        const result = await this.load(platform, open).catch(() => null);
        if (!result) break;
        for (const [key, label] of result.labels) this.cache.set(`${platform}:${key}`, label);
        const unanswered = open.filter((key) => !result.labels.has(key));
        if (!result.truncated || unanswered.length === open.length) break;
        open = unanswered;
      }
    }
  }
}
