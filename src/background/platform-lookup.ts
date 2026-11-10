import { LruCache } from '../shared/lru';
import { isLookupKey, LOOKUP_BATCH, lookupUrl, parseLookup, type LookupPlatform, type LookupResult } from '../shared/platform-labels';

const MAX_KEYS_PER_REQUEST = 200;
const TIMEOUT_MS = 10_000;

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

  constructor(private readonly load: LabelFetcher = fetchLabels) {}

  async labeled(platform: LookupPlatform, keys: string[], allowRequest: () => boolean): Promise<string[]> {
    if (!Object.hasOwn(LOOKUP_BATCH, platform) || !Array.isArray(keys)) return [];
    const size = LOOKUP_BATCH[platform];
    const labeled: string[] = [];
    const unknown: string[] = [];
    for (const key of new Set(keys.slice(0, MAX_KEYS_PER_REQUEST))) {
      if (typeof key !== 'string' || !isLookupKey(platform, key)) continue;
      const cached = this.cache.get(`${platform}:${key}`);
      if (cached === undefined) unknown.push(key);
      else if (cached) labeled.push(key);
    }
    for (let i = 0; i < unknown.length; i += size) {
      if (!allowRequest()) break;
      const batch = unknown.slice(i, i + size);
      const result = await this.load(platform, batch);
      if (!result) continue;
      for (const key of batch) {
        const label = result.labels.get(key) ?? false;
        if (label || result.complete) this.cache.set(`${platform}:${key}`, label);
        if (label) labeled.push(key);
      }
    }
    return labeled;
  }
}
