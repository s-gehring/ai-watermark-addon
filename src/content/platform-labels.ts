import type { BackgroundRequest, Platform, Verdict } from '../shared/messages';
import { commonsFilePage, deviantArtLabels, hasAiCategory, labelKeyOf, LOOKUP_BATCH, platformOf, type LookupPlatform } from '../shared/platform-labels';

const BATCH_DELAY_MS = 50;

type Label = boolean | null;
type PageLabels = () => Map<string, boolean>;
type Lookup = (keys: string[]) => Promise<Record<string, unknown>>;
type IsLarge = (image: HTMLImageElement) => boolean;

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
      const answers: unknown = await browser.runtime.sendMessage(request);
      return answers && typeof answers === 'object' ? (answers as Record<string, unknown>) : {};
    } catch {
      return {};
    }
  };
}

export function labelFirst(
  label: Promise<Verdict | null> | null,
  check: () => Promise<Verdict>,
  graceMs: number,
  show: (verdict: Verdict) => void,
): void {
  let labeled = false;
  let checking = false;
  const runCheck = () => {
    if (checking || labeled) return;
    checking = true;
    void check().then((verdict) => {
      if (!labeled) show(verdict);
    });
  };
  if (!label) return runCheck();
  const timer = window.setTimeout(runCheck, graceMs);
  void label.then((hit) => {
    window.clearTimeout(timer);
    if (!hit) return runCheck();
    labeled = true;
    show(hit);
  });
}

export class PlatformLabels {
  private readonly known = new Map<string, boolean>();
  private readonly unknown = new Set<string>();
  private readonly asked = new Map<string, Promise<Label>>();
  private queue = new Map<string, (label: Label) => void>();
  private pageView: string | null = null;
  private timer = 0;

  private constructor(
    private readonly platform: Platform,
    private readonly readPage: PageLabels | null,
    private readonly lookup: Lookup | null,
    private readonly batchSize: number,
    private readonly isLarge: IsLarge,
  ) {}

  static forPage(isLarge: IsLarge): PlatformLabels | null {
    switch (platformOf(location.hostname)) {
      case 'pixiv':
        return new PlatformLabels('pixiv', null, backgroundLookup('pixiv'), LOOKUP_BATCH.pixiv, isLarge);
      case 'deviantart':
        return new PlatformLabels('deviantart', deviantArtPageLabels, null, 0, isLarge);
      case 'commons':
        return new PlatformLabels('commons', commonsPageLabels, backgroundLookup('commons'), LOOKUP_BATCH.commons, isLarge);
      default:
        return null;
    }
  }

  verdictFor(imageUrl: string): Promise<Verdict | null> | null {
    const key = labelKeyOf(this.platform, imageUrl);
    if (!key) return null;
    return this.labelOf(key).then((ai) => (ai ? { ok: true, score: 1, source: 'platform', platform: this.platform } : null));
  }

  private labelOf(key: string): Promise<Label> {
    if (this.pageView === null) for (const [file, label] of this.readPage?.() ?? []) this.known.set(file, label);
    if (this.pageView !== location.href) this.unknown.clear();
    this.pageView = location.href;
    const label = this.known.get(key);
    if (label !== undefined) return Promise.resolve(label);
    return this.lookup && !this.unknown.has(key) ? this.ask(key) : Promise.resolve(null);
  }

  private ask(key: string): Promise<Label> {
    let answer = this.asked.get(key);
    if (!answer) {
      this.asked.set(key, (answer = new Promise((resolve) => this.enqueue(key, resolve))));
    }
    return answer;
  }

  private enqueue(key: string, resolve: (label: Label) => void): void {
    this.queue.set(key, resolve);
    this.timer ||= window.setTimeout(() => void this.flush(), BATCH_DELAY_MS);
  }

  private addOtherImages(): void {
    let room = (this.batchSize - (this.queue.size % this.batchSize)) % this.batchSize;
    for (const image of room ? document.images : []) {
      const key = labelKeyOf(this.platform, image.currentSrc || image.src);
      if (!key || this.known.has(key) || this.unknown.has(key) || this.asked.has(key) || !this.isLarge(image)) continue;
      void this.ask(key);
      if (--room === 0) break;
    }
  }

  private async flush(): Promise<void> {
    this.addOtherImages();
    this.timer = 0;
    const batch = this.queue;
    this.queue = new Map();
    const page = location.href;
    const answers = await this.lookup!([...batch.keys()]);
    for (const [key, resolve] of batch) {
      const label = answers[key];
      if (typeof label !== 'boolean' && page !== location.href) {
        this.enqueue(key, resolve);
        continue;
      }
      if (typeof label === 'boolean') this.known.set(key, label);
      else this.unknown.add(key);
      this.asked.delete(key);
      resolve(typeof label === 'boolean' ? label : null);
    }
  }
}
