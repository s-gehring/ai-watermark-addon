import { isFlagged, type BackgroundRequest, type FrameStats, type Verdict } from '../shared/messages';
import type { Settings } from '../shared/settings';
import { hasBackgroundImage, imageUrlOf, isLargeEnough, knownOriginals, largerVariantsOf } from './candidates';
import { shadowRootOf } from './dom';
import { BadgeLayer } from './overlay';

const WATCHED_ATTRIBUTES = ['src', 'srcset', 'style', 'class'];
const OBSERVED: MutationObserverInit = { childList: true, subtree: true, attributes: true, attributeFilter: WATCHED_ATTRIBUTES };
const DISCOVERY_CHUNK = 400;
const MAX_ORIGINALS = 20;

export class Scanner {
  private readonly verdicts = new Map<string, Promise<Verdict>>();
  private checkedUrl = new WeakMap<Element, string>();
  private readonly inView = new Set<Element>();
  private shadowRoots = new WeakSet<ShadowRoot>();
  private readonly layer: BadgeLayer;
  private readonly io: IntersectionObserver;
  private readonly mo: MutationObserver;
  private pendingRoots: Element[] = [];
  private discovering = false;
  private originals = 0;

  constructor(
    private settings: Settings,
    private readonly onStats: (stats: FrameStats) => void,
  ) {
    this.layer = new BadgeLayer(() => this.onStats(this.layer.stats));
    this.io = new IntersectionObserver((entries) => this.onIntersect(entries), { rootMargin: '300px' });
    this.mo = new MutationObserver((records) => this.onMutations(records));
  }

  start(): void {
    const { showScore, showChecked, threshold } = this.settings;
    this.layer.setOptions({ showScore, showChecked, threshold });
    this.shadowRoots = new WeakSet();
    this.mo.observe(document.documentElement, OBSERVED);
    document.addEventListener('load', this.onLoad, true);
    addEventListener('resize', this.onResize, { passive: true });
    this.discover(document.documentElement);
  }

  stop(): void {
    this.mo.disconnect();
    this.io.disconnect();
    document.removeEventListener('load', this.onLoad, true);
    removeEventListener('resize', this.onResize);
    this.inView.clear();
    this.pendingRoots = [];
    this.layer.clear();
  }

  restart(settings: Settings): void {
    this.stop();
    if (settings.checkMetadata !== this.settings.checkMetadata || settings.checkWatermark !== this.settings.checkWatermark) {
      this.verdicts.clear();
    }
    this.settings = settings;
    this.checkedUrl = new WeakMap();
    this.start();
  }

  private readonly onLoad = (event: Event) => {
    if (event.target instanceof HTMLImageElement && this.inView.has(event.target)) this.consider(event.target);
  };

  private readonly onResize = () => {
    for (const element of this.inView) this.consider(element);
  };

  private onIntersect(entries: IntersectionObserverEntry[]): void {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        this.inView.add(entry.target);
        this.consider(entry.target);
      } else {
        this.inView.delete(entry.target);
      }
    }
  }

  private onMutations(records: MutationRecord[]): void {
    for (const record of records) {
      if (record.type === 'childList') {
        for (const node of record.addedNodes) {
          if (node instanceof Element) this.discover(node);
        }
      } else if (record.target instanceof Element) {
        const element = record.target;
        if (this.inView.has(element)) {
          this.consider(element);
        } else if (!(element instanceof HTMLImageElement) && this.settings.includeBackgrounds) {
          this.discover(element, false);
        }
      }
    }
    this.layer.scheduleUpdate();
  }

  private discover(root: Element, deep = true): void {
    if (deep) {
      this.pendingRoots.push(root);
    } else {
      this.observeIfCandidate(root);
      return;
    }
    if (this.discovering) return;
    this.discovering = true;
    const step = (deadline?: IdleDeadline) => {
      let budget = DISCOVERY_CHUNK;
      while (this.pendingRoots.length && budget > 0 && (!deadline || deadline.timeRemaining() > 1)) {
        const next = this.pendingRoots.pop()!;
        this.observeIfCandidate(next);
        budget--;
        const shadow = shadowRootOf(next);
        if (shadow) this.watchShadowRoot(shadow);
        for (const parent of shadow ? [shadow, next] : [next]) {
          for (let child = parent.lastElementChild; child; child = child.previousElementSibling) {
            this.pendingRoots.push(child);
          }
        }
      }
      if (this.pendingRoots.length) {
        requestIdleCallback(step, { timeout: 500 });
      } else {
        this.discovering = false;
      }
    };
    step();
  }

  private watchShadowRoot(root: ShadowRoot): void {
    if (this.shadowRoots.has(root)) return;
    this.shadowRoots.add(root);
    this.mo.observe(root, OBSERVED);
    root.addEventListener('load', this.onLoad, true);
  }

  private observeIfCandidate(element: Element): void {
    if (element instanceof HTMLImageElement) {
      this.io.observe(element);
    } else if (this.settings.includeBackgrounds && hasBackgroundImage(element)) {
      this.io.observe(element);
    }
  }

  private consider(element: Element): void {
    const url = imageUrlOf(element, this.settings.includeBackgrounds);
    if (!url || !isLargeEnough(element.getBoundingClientRect(), this.settings.minSize)) {
      this.checkedUrl.delete(element);
      this.layer.remove(element);
      return;
    }
    if (this.checkedUrl.get(element) === url) return;
    this.checkedUrl.set(element, url);
    this.layer.set(element, { kind: 'pending' });
    void this.verdictFor(url, element).then((verdict) => {
      if (this.checkedUrl.get(element) !== url) return;
      if (!verdict.ok) {
        this.layer.set(element, { kind: 'failed', error: verdict.error });
      } else if (verdict.source === 'model' && !isFlagged(verdict, this.settings.threshold)) {
        this.layer.set(element, { kind: 'clean', verdict });
      } else {
        this.layer.set(element, { kind: 'ai', verdict });
      }
    });
  }

  private verdictFor(url: string, element: Element): Promise<Verdict> {
    let verdict = this.verdicts.get(url);
    if (!verdict) {
      verdict = requestVerdict(url, element, this.originalFor(url, element)).then((result) => {
        if (!result.ok && result.error !== 'unsupported' && result.error !== 'decode') this.verdicts.delete(url);
        return result;
      });
      this.verdicts.set(url, verdict);
    }
    return verdict;
  }

  private originalFor(url: string, element: Element): string[] | undefined {
    const { checkMetadata, checkWatermark } = this.settings;
    if (this.originals >= MAX_ORIGINALS || !(checkMetadata || checkWatermark)) return undefined;
    const originals = element instanceof HTMLImageElement ? largerVariantsOf(element, url) : knownOriginals(url);
    if (!originals.length) return undefined;
    this.originals++;
    return originals;
  }
}

async function send(request: BackgroundRequest): Promise<Verdict> {
  try {
    return (await browser.runtime.sendMessage(request)) as Verdict;
  } catch {
    return { ok: false, error: 'model' };
  }
}

async function requestVerdict(url: string, element: Element, original?: string[]): Promise<Verdict> {
  const verdict = url.startsWith('blob:') ? await classifyBlob(url) : await send({ type: 'classify', url, original });
  if (verdict.ok || verdict.error !== 'fetch' || !(element instanceof HTMLImageElement)) return verdict;
  const bytes = await pixelsOf(element);
  return bytes ? send({ type: 'classify', url, bytes, original }) : verdict;
}

async function classifyBlob(url: string): Promise<Verdict> {
  const bytes = await fetch(url)
    .then((r) => r.arrayBuffer())
    .catch(() => null);
  return bytes ? send({ type: 'classify', url, bytes }) : { ok: false, error: 'fetch' };
}

async function pixelsOf(image: HTMLImageElement): Promise<ArrayBuffer | null> {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.getContext('2d')!.drawImage(image, 0, 0);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
    return blob ? await blob.arrayBuffer() : null;
  } catch {
    return null;
  }
}
