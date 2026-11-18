import { CANCELLED, isFlagged, isPossible, type BackgroundRequest, type FrameStats, type Verdict } from '../shared/messages';
import { possibleFrom, type Settings } from '../shared/settings';
import { hasBackgroundImage, imageUrlOf, isLargeEnough, knownOriginals, largerVariantsOf } from './candidates';
import { shadowRootOf } from './dom';
import { BadgeLayer } from './overlay';
import { labelFirst, PlatformLabels } from './platform-labels';

const WATCHED_ATTRIBUTES = ['src', 'srcset', 'style', 'class'];
const OBSERVED: MutationObserverInit = { childList: true, subtree: true, attributes: true, attributeFilter: WATCHED_ATTRIBUTES };
const DISCOVERY_CHUNK = 400;
const LABEL_GRACE_MS = 800;
const OUT_OF_VIEW_DELAY_MS = 100;

export function mayContinue(deadline: IdleDeadline | undefined): boolean {
  return !deadline || deadline.didTimeout || deadline.timeRemaining() > 1;
}

export function outOfView(awaited: ReadonlyMap<string, number>, viewed: Iterable<string | undefined>): number[] {
  const unseen = new Map(awaited);
  for (const url of viewed) {
    if (url !== undefined) unseen.delete(url);
  }
  return [...unseen.values()];
}

export class Scanner {
  private readonly verdicts = new Map<string, Promise<Verdict>>();
  private readonly awaited = new Map<string, number>();
  private reported = new Set<number>();
  private requests = 0;
  private reportTimer = 0;
  private checkedUrl = new WeakMap<Element, string>();
  private readonly inView = new Set<Element>();
  private shadowRoots = new WeakSet<ShadowRoot>();
  private readonly platformLabels = PlatformLabels.forPage((image) => isLargeEnough(image.getBoundingClientRect(), this.settings.minSize));
  private readonly layer: BadgeLayer;
  private readonly io: IntersectionObserver;
  private readonly mo: MutationObserver;
  private pendingRoots: Element[] = [];
  private discovering = false;

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
    this.layer.setOptions({ showScore, showChecked, threshold, possibleFrom: possibleFrom(this.settings) });
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
    clearTimeout(this.reportTimer);
    this.reportTimer = 0;
    this.checkedUrl = new WeakMap();
    this.inView.clear();
    this.pendingRoots = [];
    this.layer.clear();
  }

  restart(settings: Settings): void {
    this.stop();
    if (settings.checkMetadata !== this.settings.checkMetadata || settings.checkWatermark !== this.settings.checkWatermark) {
      this.verdicts.clear();
      this.dropRequests();
    }
    this.settings = settings;
    this.start();
  }

  leave(): void {
    this.stop();
    this.dropRequests();
  }

  private dropRequests(): void {
    if (!this.awaited.size) return;
    for (const url of this.awaited.keys()) this.verdicts.delete(url);
    this.awaited.clear();
    this.reported.clear();
    notify({ type: 'drop-requests' });
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
    this.scheduleReport();
  }

  private scheduleReport(): void {
    if (this.reportTimer || !this.awaited.size) return;
    this.reportTimer = window.setTimeout(() => {
      this.reportTimer = 0;
      if (this.discovering) {
        this.scheduleReport();
        return;
      }
      const ids = outOfView(
        this.awaited,
        Array.from(this.inView, (element) => this.checkedUrl.get(element)),
      );
      if (ids.length === this.reported.size && ids.every((id) => this.reported.has(id))) return;
      this.reported = new Set(ids);
      notify({ type: 'out-of-view', ids });
    }, OUT_OF_VIEW_DELAY_MS);
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
      while (this.pendingRoots.length && budget > 0 && mayContinue(deadline)) {
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
      this.forget(element);
      return;
    }
    if (this.checkedUrl.get(element) === url) return;
    this.checkedUrl.set(element, url);
    this.scheduleReport();
    this.layer.set(element, { kind: 'pending' });
    const label = this.settings.checkPlatform ? (this.platformLabels?.verdictFor(url) ?? null) : null;
    const check = () => (this.checkedUrl.get(element) === url ? this.verdictFor(url, element) : Promise.resolve(CANCELLED));
    labelFirst(label, check, LABEL_GRACE_MS, (verdict) => {
      if (this.checkedUrl.get(element) !== url) return;
      if (!verdict.ok) {
        if (verdict.error === 'cancelled') {
          this.forget(element);
          if (this.inView.has(element)) this.consider(element);
        } else {
          this.layer.set(element, { kind: 'failed', error: verdict.error });
        }
      } else if (verdict.source === 'model' && !isFlagged(verdict, this.settings.threshold)) {
        const possible = isPossible(verdict, this.settings.threshold, possibleFrom(this.settings));
        this.layer.set(element, { kind: possible ? 'possible' : 'clean', verdict });
      } else {
        this.layer.set(element, { kind: 'ai', verdict });
      }
    });
  }

  private forget(element: Element): void {
    this.checkedUrl.delete(element);
    this.layer.remove(element);
    this.scheduleReport();
  }

  private verdictFor(url: string, element: Element): Promise<Verdict> {
    const known = this.verdicts.get(url);
    if (known) return known;
    const id = this.requests++;
    const verdict = requestVerdict(id, url, element, this.originalFor(url, element)).then((result) => {
      if (this.awaited.get(url) === id) this.awaited.delete(url);
      this.reported.delete(id);
      const final = result.ok || result.error === 'unsupported' || result.error === 'decode';
      if (!final && this.verdicts.get(url) === verdict) this.verdicts.delete(url);
      return result;
    });
    this.verdicts.set(url, verdict);
    this.awaited.set(url, id);
    this.scheduleReport();
    return verdict;
  }

  private originalFor(url: string, element: Element): string[] | undefined {
    const { checkMetadata, checkWatermark } = this.settings;
    if (!(checkMetadata || checkWatermark)) return undefined;
    const originals = element instanceof HTMLImageElement ? largerVariantsOf(element, url) : knownOriginals(url);
    return originals.length ? originals : undefined;
  }
}

async function send(request: BackgroundRequest): Promise<Verdict> {
  try {
    return (await browser.runtime.sendMessage(request)) as Verdict;
  } catch {
    return { ok: false, error: 'model' };
  }
}

function notify(request: BackgroundRequest): void {
  void browser.runtime.sendMessage(request).catch(() => {});
}

async function requestVerdict(id: number, url: string, element: Element, original?: string[]): Promise<Verdict> {
  const page = location.href;
  const verdict = url.startsWith('blob:') ? await classifyBlob(id, url, page) : await send({ type: 'classify', id, url, page, original });
  if (verdict.ok || verdict.error !== 'fetch' || !(element instanceof HTMLImageElement)) return verdict;
  const bytes = await pixelsOf(element);
  return bytes ? send({ type: 'classify', id, url, page, bytes, original }) : verdict;
}

async function classifyBlob(id: number, url: string, page: string): Promise<Verdict> {
  const bytes = await fetch(url)
    .then((r) => r.arrayBuffer())
    .catch(() => null);
  return bytes ? send({ type: 'classify', id, url, page, bytes }) : { ok: false, error: 'fetch' };
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
