import type { BackgroundRequest, Platform, Verdict } from '../shared/messages';
import { commonsFilePage, deviantArtLabels, hasAiCategory, labelKeyOf, LOOKUP_BATCH, platformOf, type LookupPlatform } from '../shared/platform-labels';

const BATCH_DELAY_MS = 150;

type PageLabels = () => Map<string, boolean>;
type Lookup = (keys: string[]) => Promise<string[]>;
type ImageUrls = () => string[];

function deviantArtPageLabels(): Map<string, boolean> {
  for (const script of document.scripts) {
    const text = script.src ? '' : (script.textContent ?? '');
    if (text.includes('__INITIAL_STATE__')) return deviantArtLabels(text);
  }
  return new Map();
}

function commonsPageLabels(): Map<string, boolean> {
  const labels = new Map<string, boolean>();
  const title = commonsFilePage(location.hostname, location.pathname);
  const categories = document.querySelectorAll('#catlinks li a');
  if (title && categories.length) labels.set(title, hasAiCategory([...categories].map((link) => link.textContent ?? '')));
  return labels;
}

function backgroundLookup(platform: LookupPlatform): Lookup {
  return async (keys) => {
    const request: BackgroundRequest = { type: 'platform-labels', platform, keys, page: location.href };
    try {
      const labeled: unknown = await browser.runtime.sendMessage(request);
      return Array.isArray(labeled) ? (labeled as string[]) : [];
    } catch {
      return [];
    }
  };
}

export class PlatformLabels {
  private readonly labels = new Map<string, Promise<boolean>>();
  private readonly waiting = new Map<string, (labeled: boolean) => void>();
  private fromPage: Map<string, boolean> | null = null;
  private timer = 0;

  private constructor(
    private readonly platform: Platform,
    private readonly readPage: PageLabels | null,
    private readonly lookup: Lookup | null,
    private readonly batchSize: number,
    private readonly otherImages: ImageUrls,
  ) {}

  static forPage(otherImages: ImageUrls): PlatformLabels | null {
    switch (platformOf(location.hostname)) {
      case 'pixiv':
        return new PlatformLabels('pixiv', null, backgroundLookup('pixiv'), LOOKUP_BATCH.pixiv, otherImages);
      case 'deviantart':
        return new PlatformLabels('deviantart', deviantArtPageLabels, null, 0, otherImages);
      case 'commons':
        return new PlatformLabels('commons', commonsPageLabels, backgroundLookup('commons'), LOOKUP_BATCH.commons, otherImages);
      default:
        return null;
    }
  }

  verdictFor(imageUrl: string): Promise<Verdict | null> | null {
    const key = labelKeyOf(this.platform, imageUrl);
    if (!key) return null;
    let labeled = this.labels.get(key);
    if (!labeled) this.labels.set(key, (labeled = this.resolve(key)));
    return labeled.then((ai) => (ai ? { ok: true, score: 1, source: 'platform', platform: this.platform } : null));
  }

  private resolve(key: string): Promise<boolean> {
    this.fromPage ??= this.readPage?.() ?? new Map();
    const known = this.fromPage.get(key);
    if (known !== undefined || !this.lookup) return Promise.resolve(known ?? false);
    this.timer ||= window.setTimeout(() => void this.flush(), BATCH_DELAY_MS);
    return this.wait(key);
  }

  private wait(key: string): Promise<boolean> {
    return new Promise((resolve) => this.waiting.set(key, resolve));
  }

  private addOtherImages(): void {
    let room = (this.batchSize - (this.waiting.size % this.batchSize)) % this.batchSize;
    for (const url of room ? this.otherImages() : []) {
      const key = labelKeyOf(this.platform, url);
      if (!key || this.labels.has(key) || this.fromPage?.has(key)) continue;
      this.labels.set(key, this.wait(key));
      if (--room === 0) break;
    }
  }

  private async flush(): Promise<void> {
    this.timer = 0;
    this.addOtherImages();
    const batch = new Map(this.waiting);
    this.waiting.clear();
    const labeled = new Set(await this.lookup!([...batch.keys()]));
    for (const [key, resolve] of batch) resolve(labeled.has(key));
  }
}
