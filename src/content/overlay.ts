import type { Verdict } from '../shared/messages';

const INSET = 6;

const STYLE = `
:host { all: initial; }
.layer { position: fixed; top: 0; left: 0; width: 0; height: 0; overflow: visible; pointer-events: none; }
.badge {
  position: absolute; top: 0; left: 0;
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 7px; border-radius: 999px;
  background: rgba(91, 33, 182, 0.9); color: #fff;
  border: 1px solid rgba(255, 255, 255, 0.45);
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
  font: 700 11px/1.2 system-ui, -apple-system, "Segoe UI", sans-serif; letter-spacing: 0.02em;
  white-space: nowrap; cursor: pointer; pointer-events: auto; user-select: none;
  backdrop-filter: blur(2px);
}
.badge[hidden] { display: none; }
.score { font-weight: 500; opacity: 0.85; }
`;

interface Badge {
  node: HTMLElement;
  dismissed: boolean;
}

export class BadgeLayer {
  private readonly host = document.createElement('ai-watermark-layer');
  private readonly root: HTMLElement;
  private readonly badges = new Map<Element, Badge>();
  private frame = 0;

  constructor(private readonly onChange: () => void) {
    const shadow = this.host.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = STYLE;
    this.root = document.createElement('div');
    this.root.className = 'layer';
    shadow.append(style, this.root);

    const schedule = () => this.scheduleUpdate();
    addEventListener('scroll', schedule, { capture: true, passive: true });
    addEventListener('resize', schedule, { passive: true });
    setInterval(() => this.badges.size && this.scheduleUpdate(), 750);
  }

  get count(): number {
    let visible = 0;
    for (const badge of this.badges.values()) if (!badge.dismissed) visible++;
    return visible;
  }

  show(element: Element, verdict: Verdict, showScore: boolean): void {
    if (!verdict.ok) return;
    const existing = this.badges.get(element);
    const node = existing?.node ?? document.createElement('div');
    node.className = 'badge';
    node.replaceChildren(label(verdict, showScore));
    node.title = tooltip(verdict);
    if (!existing) {
      node.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        const badge = this.badges.get(element);
        if (badge) badge.dismissed = true;
        node.hidden = true;
        this.onChange();
      });
      this.root.append(node);
      this.badges.set(element, { node, dismissed: false });
      this.onChange();
    }
    this.scheduleUpdate();
  }

  hide(element: Element): void {
    const badge = this.badges.get(element);
    if (!badge) return;
    badge.node.remove();
    this.badges.delete(element);
    this.onChange();
  }

  clear(): void {
    for (const element of [...this.badges.keys()]) this.hide(element);
  }

  scheduleUpdate(): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.update();
    });
  }

  private update(): void {
    if (!this.host.isConnected) document.documentElement.append(this.host);
    const viewportWidth = document.documentElement.clientWidth || innerWidth;
    const viewportHeight = document.documentElement.clientHeight || innerHeight;
    for (const [element, badge] of [...this.badges]) {
      if (!element.isConnected) {
        this.hide(element);
        continue;
      }
      if (badge.dismissed) continue;
      const rect = element.getBoundingClientRect();
      const width = badge.node.offsetWidth || 40;
      const height = badge.node.offsetHeight || 20;
      const right = Math.min(rect.right, viewportWidth);
      const top = Math.max(rect.top, 0);
      const fits = right - Math.max(rect.left, 0) >= width + 2 * INSET && rect.bottom - top >= height + 2 * INSET;
      const visible =
        fits &&
        rect.bottom > 0 &&
        rect.top < viewportHeight &&
        rect.right > 0 &&
        rect.left < viewportWidth &&
        isRendered(element);
      badge.node.hidden = !visible;
      if (visible) {
        badge.node.style.transform = `translate(${Math.round(right - INSET - width)}px, ${Math.round(top + INSET)}px)`;
      }
    }
  }
}

function isRendered(element: Element): boolean {
  if (typeof element.checkVisibility === 'function') {
    return element.checkVisibility({ opacityProperty: true, visibilityProperty: true });
  }
  return true;
}

function label(verdict: Extract<Verdict, { ok: true }>, showScore: boolean): DocumentFragment {
  const fragment = document.createDocumentFragment();
  fragment.append('AI');
  let detail: string | null = null;
  if (verdict.source === 'metadata') detail = browser.i18n.getMessage('badgeMetadata');
  else if (showScore) detail = `${Math.round(verdict.score * 100)} %`;
  if (detail) {
    const span = document.createElement('span');
    span.className = 'score';
    span.textContent = detail;
    fragment.append(span);
  }
  return fragment;
}

function tooltip(verdict: Extract<Verdict, { ok: true }>): string {
  const hint = browser.i18n.getMessage('tooltipDismiss');
  if (verdict.source === 'metadata') {
    return `${browser.i18n.getMessage(`reason_${verdict.reason.replace('-', '_')}`)}\n${hint}`;
  }
  return `${browser.i18n.getMessage('tooltipModel', [String(Math.round(verdict.score * 100))])}\n${hint}`;
}
