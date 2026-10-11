import { EMPTY_STATS, type FrameStats, type Verdict } from '../shared/messages';

const INSET = 6;
const SVG_NS = 'http://www.w3.org/2000/svg';

const STYLE = `
:host { all: initial; }
.layer { position: fixed; top: 0; left: 0; width: 0; height: 0; overflow: visible; pointer-events: none; }
.badge {
  position: absolute; top: 0; left: 0;
  display: inline-flex; align-items: center; justify-content: center; gap: 4px;
  box-sizing: border-box; height: 22px; border-radius: 999px;
  color: #fff; border: 1px solid rgba(255, 255, 255, 0.45);
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
  font: 700 11px/1 system-ui, -apple-system, "Segoe UI", sans-serif; letter-spacing: 0.02em;
  white-space: nowrap; cursor: pointer; pointer-events: auto; user-select: none;
}
.badge[hidden] { display: none; }
.ai { padding: 0 8px; background: rgba(91, 33, 182, 0.92); }
.clean, .pending, .failed { width: 22px; }
.clean { background: rgba(21, 128, 61, 0.88); }
.failed { background: rgba(82, 82, 91, 0.85); font-size: 12px; }
.pending { background: rgba(24, 24, 27, 0.6); }
.score { font-weight: 500; opacity: 0.85; }
.spinner {
  width: 10px; height: 10px; border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.3); border-top-color: #fff;
  animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .spinner { animation: none; border-color: #fff; opacity: 0.7; } }
`;

export type BadgeState =
  | { kind: 'pending' }
  | { kind: 'ai'; verdict: Extract<Verdict, { ok: true }> }
  | { kind: 'clean'; verdict: Extract<Verdict, { ok: true; source: 'model' }> }
  | { kind: 'failed' };

export interface DisplayOptions {
  showScore: boolean;
  showChecked: boolean;
}

interface Badge {
  node: HTMLElement;
  state: BadgeState;
  dismissed: boolean;
}

export class BadgeLayer {
  private readonly host = document.createElement('ai-watermark-layer');
  private readonly root: HTMLElement;
  private readonly badges = new Map<Element, Badge>();
  private options: DisplayOptions = { showScore: true, showChecked: true };
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

  get stats(): FrameStats {
    const stats = { ...EMPTY_STATS };
    for (const { state } of this.badges.values()) {
      if (state.kind === 'ai') stats.flagged++;
      else stats[state.kind]++;
    }
    return stats;
  }

  setOptions(options: DisplayOptions): void {
    this.options = options;
    for (const badge of this.badges.values()) render(badge, options);
    this.scheduleUpdate();
  }

  set(element: Element, state: BadgeState): void {
    let badge = this.badges.get(element);
    if (!badge) {
      const node = document.createElement('div');
      badge = { node, state, dismissed: false };
      const created = badge;
      node.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        created.dismissed = true;
        node.hidden = true;
      });
      this.root.append(node);
      this.badges.set(element, badge);
    }
    badge.state = state;
    badge.dismissed = false;
    render(badge, this.options);
    this.onChange();
    this.scheduleUpdate();
  }

  remove(element: Element): void {
    const badge = this.badges.get(element);
    if (!badge) return;
    badge.node.remove();
    this.badges.delete(element);
    this.onChange();
  }

  clear(): void {
    for (const element of [...this.badges.keys()]) this.remove(element);
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
        this.remove(element);
        continue;
      }
      if (badge.dismissed || !isDisplayed(badge.state, this.options)) continue;
      const rect = element.getBoundingClientRect();
      const width = badge.node.offsetWidth || 40;
      const height = badge.node.offsetHeight || 22;
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

function isDisplayed(state: BadgeState, options: DisplayOptions): boolean {
  return state.kind === 'ai' || options.showChecked;
}

function isRendered(element: Element): boolean {
  if (typeof element.checkVisibility === 'function') {
    return element.checkVisibility({ opacityProperty: true, visibilityProperty: true });
  }
  return true;
}

function render(badge: Badge, options: DisplayOptions): void {
  const { node, state } = badge;
  node.className = `badge ${state.kind}`;
  node.hidden = badge.dismissed || !isDisplayed(state, options);
  node.title = tooltip(state);
  switch (state.kind) {
    case 'ai':
      node.replaceChildren(aiLabel(state.verdict, options.showScore));
      break;
    case 'clean':
      node.replaceChildren(checkMark());
      break;
    case 'pending': {
      const spinner = document.createElement('span');
      spinner.className = 'spinner';
      node.replaceChildren(spinner);
      break;
    }
    case 'failed':
      node.replaceChildren('?');
      break;
  }
}

function checkMark(): SVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('width', '12');
  svg.setAttribute('height', '12');
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', 'M3 8.5l3.2 3.2L13 4.8');
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', '#fff');
  path.setAttribute('stroke-width', '2.4');
  path.setAttribute('stroke-linecap', 'round');
  path.setAttribute('stroke-linejoin', 'round');
  svg.append(path);
  return svg;
}

function aiLabel(verdict: Extract<Verdict, { ok: true }>, showScore: boolean): DocumentFragment {
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

function tooltip(state: BadgeState): string {
  const hint = browser.i18n.getMessage('tooltipDismiss');
  switch (state.kind) {
    case 'pending':
      return browser.i18n.getMessage('tooltipPending');
    case 'failed':
      return `${browser.i18n.getMessage('tooltipFailed')}\n${hint}`;
    case 'clean':
      return `${browser.i18n.getMessage('tooltipClean', [String(Math.round(state.verdict.score * 100))])}\n${hint}`;
    case 'ai':
      if (state.verdict.source === 'metadata') {
        return `${browser.i18n.getMessage(`reason_${state.verdict.reason.replace('-', '_')}`)}\n${hint}`;
      }
      return `${browser.i18n.getMessage('tooltipModel', [String(Math.round(state.verdict.score * 100))])}\n${hint}`;
  }
}
