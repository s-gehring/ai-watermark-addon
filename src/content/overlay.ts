import model from '../../model.json';
import { EMPTY_STATS, isFlagged, type CheckStatus, type FrameStats, type Verdict, type VerdictError, type WatermarkScheme } from '../shared/messages';
import { composedContains, LAYER_TAG, parentOf } from './dom';

const INSET = 6;
const GAP = 6;
const MARGIN = 8;
const CLOSE_DELAY = 200;
const SVG_NS = 'http://www.w3.org/2000/svg';

const STYLE = `
:host { all: initial; }
.layer { position: absolute; top: 0; left: 0; z-index: 2147483647; width: 0; height: 0; overflow: visible; pointer-events: none; }
.layer.fixed { position: fixed; }
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
.details {
  position: absolute; top: 0; left: 0;
  box-sizing: border-box; width: 300px; max-width: calc(100vw - ${2 * MARGIN}px);
  padding: 10px 12px; border-radius: 10px;
  background: rgba(24, 24, 27, 0.96); color: #f4f4f5;
  border: 1px solid rgba(255, 255, 255, 0.14);
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
  font: 400 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  text-align: left; cursor: default; pointer-events: auto; user-select: text;
}
.details[hidden] { display: none; }
.title { margin-bottom: 8px; font-size: 13px; font-weight: 700; }
.checks { display: grid; gap: 6px; margin: 0; padding: 0; list-style: none; }
.check { display: grid; grid-template-columns: 8px 1fr auto; column-gap: 8px; align-items: baseline; }
.dot { width: 8px; height: 8px; border-radius: 50%; align-self: center; }
.hit .dot { background: #a78bfa; }
.clear .dot { background: #4ade80; }
.off .dot { background: #71717a; }
.value { text-align: right; font-variant-numeric: tabular-nums; }
.off .value, .note { color: #a1a1aa; }
.check > .note, .models { grid-column: 2 / -1; }
.models { display: grid; grid-template-columns: 1fr 48px auto; gap: 4px 8px; align-items: center; margin-top: 4px; }
.models > .note { grid-column: 1 / -1; }
.bar { position: relative; height: 4px; border-radius: 2px; background: rgba(255, 255, 255, 0.15); }
.bar > span { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 2px; background: #e4e4e7; }
.bar::after { content: ""; position: absolute; top: -2px; bottom: -2px; left: var(--threshold); width: 1px; background: #fbbf24; }
.hint { margin-top: 10px; }
`;

type OkVerdict = Extract<Verdict, { ok: true }>;
type ModelVerdict = Extract<Verdict, { ok: true; source: 'model' }>;

export type BadgeState =
  | { kind: 'pending' }
  | { kind: 'ai'; verdict: OkVerdict }
  | { kind: 'clean'; verdict: ModelVerdict }
  | { kind: 'failed'; error: VerdictError };

export interface DisplayOptions {
  showScore: boolean;
  showChecked: boolean;
  threshold: number;
}

interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

interface Badge {
  node: HTMLElement;
  state: BadgeState;
  dismissed: boolean;
}

export class BadgeLayer {
  private readonly host = document.createElement(LAYER_TAG);
  private readonly page: HTMLElement;
  private readonly screen: HTMLElement;
  private readonly details: HTMLElement;
  private readonly badges = new Map<Element, Badge>();
  private options: DisplayOptions = { showScore: true, showChecked: true, threshold: model.threshold };
  private open: Element | null = null;
  private closeTimer = 0;
  private frame = 0;

  constructor(private readonly onChange: () => void) {
    const shadow = this.host.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = STYLE;
    this.page = text('div', 'layer');
    this.screen = text('div', 'layer fixed');
    this.details = text('div', 'details');
    this.details.hidden = true;
    this.details.addEventListener('pointerenter', () => clearTimeout(this.closeTimer));
    this.details.addEventListener('pointerleave', () => this.scheduleClose());
    this.page.append(this.details);
    shadow.append(style, this.page, this.screen);
    for (const type of ['click', 'pointerdown', 'mousedown']) {
      for (const layer of [this.page, this.screen]) layer.addEventListener(type, (event) => event.stopPropagation());
    }

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
    this.renderDetails();
    this.scheduleUpdate();
  }

  set(element: Element, state: BadgeState): void {
    let badge = this.badges.get(element);
    if (!badge) {
      const node = document.createElement('div');
      node.setAttribute('role', 'button');
      badge = { node, state, dismissed: false };
      const created = badge;
      node.addEventListener('click', (event) => {
        event.preventDefault();
        this.close();
        created.dismissed = true;
        node.hidden = true;
      });
      node.addEventListener('pointerenter', () => this.show(element));
      node.addEventListener('pointerleave', () => this.scheduleClose());
      this.place(node, this.page);
      this.badges.set(element, badge);
    }
    badge.state = state;
    badge.dismissed = false;
    render(badge, this.options);
    if (element === this.open) this.renderDetails();
    this.onChange();
    this.scheduleUpdate();
  }

  remove(element: Element): void {
    const badge = this.badges.get(element);
    if (!badge) return;
    if (element === this.open) this.close();
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

  private show(element: Element): void {
    clearTimeout(this.closeTimer);
    const badge = this.badges.get(element);
    if (!badge || this.open === element) return;
    this.open = element;
    this.renderDetails();
    this.details.hidden = false;
    this.placeDetails(badge.node);
  }

  private place(node: HTMLElement, layer: HTMLElement): void {
    if (node.parentElement !== layer) layer.insertBefore(node, this.details.parentElement === layer ? this.details : null);
  }

  private scheduleClose(): void {
    clearTimeout(this.closeTimer);
    this.closeTimer = window.setTimeout(() => this.close(), CLOSE_DELAY);
  }

  private close(): void {
    clearTimeout(this.closeTimer);
    this.open = null;
    this.details.hidden = true;
  }

  private renderDetails(): void {
    const badge = this.open && this.badges.get(this.open);
    if (!badge) return;
    const hint = text('div', 'note hint', browser.i18n.getMessage('tooltipDismiss'));
    this.details.replaceChildren(...detailsContent(badge.state, this.options.threshold), hint);
  }

  private placeDetails(anchor: HTMLElement): void {
    const layer = anchor.parentElement!;
    if (this.details.parentElement !== layer) layer.append(this.details);
    const origin = layer.getBoundingClientRect();
    const viewportWidth = document.documentElement.clientWidth || innerWidth;
    const viewportHeight = document.documentElement.clientHeight || innerHeight;
    const badge = anchor.getBoundingClientRect();
    const width = this.details.offsetWidth;
    const height = this.details.offsetHeight;
    const x = Math.min(Math.max(badge.right - width, MARGIN), viewportWidth - width - MARGIN);
    const below = badge.bottom + GAP;
    const y = below + height <= viewportHeight - MARGIN ? below : Math.max(badge.top - GAP - height, MARGIN);
    this.details.style.transform = `translate(${Math.round(x - origin.left)}px, ${Math.round(y - origin.top)}px)`;
  }

  private update(): void {
    if (!this.host.isConnected) document.documentElement.append(this.host);
    const viewportWidth = document.documentElement.clientWidth || innerWidth;
    const viewportHeight = document.documentElement.clientHeight || innerHeight;
    const origins = new Map([this.page, this.screen].map((layer) => [layer, layer.getBoundingClientRect()]));
    for (const [element, badge] of [...this.badges]) {
      if (!element.isConnected) {
        this.remove(element);
        continue;
      }
      if (badge.dismissed || !isDisplayed(badge.state, this.options)) continue;
      const width = badge.node.offsetWidth || 40;
      const height = badge.node.offsetHeight || 22;
      const clip = visibleBox(element, viewportWidth, viewportHeight);
      const box = clip?.box;
      const x = box ? box.right - INSET - width : 0;
      const y = box ? box.top + INSET : 0;
      const visible =
        !!box &&
        box.right - box.left >= width + 2 * INSET &&
        box.bottom - box.top >= height + 2 * INSET &&
        isRendered(element) &&
        !this.isCovered(element, x + width / 2, y + height / 2);
      badge.node.hidden = !visible;
      if (!visible || !clip) continue;
      const layer = clip.fixed ? this.screen : this.page;
      const origin = origins.get(layer)!;
      this.place(badge.node, layer);
      badge.node.style.transform = `translate(${Math.round(x - origin.left)}px, ${Math.round(y - origin.top)}px)`;
    }
    const open = this.open && this.badges.get(this.open);
    if (open && open.node.hidden) this.close();
    else if (open) this.placeDetails(open.node);
  }

  private isCovered(element: Element, x: number, y: number): boolean {
    for (const hit of document.elementsFromPoint(x, y)) {
      if (hit === this.host) continue;
      if (composedContains(hit, element) || composedContains(element, hit)) return false;
      for (let node: Element | null = hit; node && !composedContains(node, element); node = parentOf(node)) {
        const { position } = getComputedStyle(node);
        if (position === 'fixed' || position === 'sticky') return true;
      }
    }
    return false;
  }
}

function visibleBox(element: Element, viewportWidth: number, viewportHeight: number): { box: Box; fixed: boolean } | null {
  const rect = element.getBoundingClientRect();
  if (rect.bottom < -viewportHeight || rect.top > 2 * viewportHeight) return null;
  const box: Box = {
    left: Math.max(rect.left, 0),
    top: rect.top,
    right: Math.min(rect.right, viewportWidth),
    bottom: rect.bottom,
  };
  let fixed = false;
  for (let node = parentOf(element); node && node !== document.body; node = parentOf(node)) {
    if (box.right <= box.left || box.bottom <= box.top) return null;
    const style = getComputedStyle(node);
    if (style.overflowX !== 'visible' || style.overflowY !== 'visible') {
      const clip = node.getBoundingClientRect();
      if (style.overflowX !== 'visible') {
        box.left = Math.max(box.left, clip.left);
        box.right = Math.min(box.right, clip.right);
      }
      if (style.overflowY !== 'visible') {
        box.top = Math.max(box.top, clip.top);
        box.bottom = Math.min(box.bottom, clip.bottom);
      }
    }
    if (style.position === 'fixed') {
      fixed = true;
      box.top = Math.max(box.top, 0);
      box.bottom = Math.min(box.bottom, viewportHeight);
      break;
    }
  }
  return box.right > box.left && box.bottom > box.top ? { box, fixed } : null;
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

function aiLabel(verdict: OkVerdict, showScore: boolean): DocumentFragment {
  const fragment = document.createDocumentFragment();
  fragment.append('AI');
  let detail: string | null = null;
  if (verdict.source === 'synthid' || (verdict.source === 'metadata' && verdict.reason === 'synthid')) detail = 'SynthID';
  else if (verdict.source === 'watermark') detail = browser.i18n.getMessage('badgeWatermark');
  else if (verdict.source === 'metadata') detail = browser.i18n.getMessage('badgeMetadata');
  else if (showScore) detail = `${Math.round(verdict.score * 100)} %`;
  if (detail) {
    const span = document.createElement('span');
    span.className = 'score';
    span.textContent = detail;
    fragment.append(span);
  }
  return fragment;
}

const SCHEME_NAMES: Record<WatermarkScheme, string> = {
  'stable-diffusion': 'Stable Diffusion',
  sdxl: 'SDXL',
  flux: 'FLUX',
  'nova-canvas': 'Amazon Nova Canvas',
  bytedance: 'ByteDance (Doubao, Jimeng)',
  'gemini-visible': 'Gemini',
  'kling-visible': 'Kling',
  'grok-visible': 'Grok',
  'doubao-visible': 'Doubao',
  'jimeng-visible': 'Jimeng',
  'qwen-visible': 'Qwen',
  'bing-visible': 'Bing Image Creator',
  'meta-visible': 'Meta AI',
  'dalle-visible': 'DALL·E 2',
};

const VISIBLE_SCHEMES: ReadonlySet<WatermarkScheme> = new Set([
  'gemini-visible',
  'kling-visible',
  'grok-visible',
  'doubao-visible',
  'jimeng-visible',
  'qwen-visible',
  'bing-visible',
  'meta-visible',
  'dalle-visible',
]);

function reasonText(verdict: Extract<Verdict, { source: 'metadata' }>): string {
  return browser.i18n.getMessage(`reason_${verdict.reason.replace('-', '_')}`);
}

function detailsContent(state: BadgeState, threshold: number): HTMLElement[] {
  switch (state.kind) {
    case 'pending':
      return [text('div', 'title', browser.i18n.getMessage('tooltipPending'))];
    case 'failed':
      return [
        text('div', 'title', browser.i18n.getMessage('detailsFailed')),
        text('div', 'note', browser.i18n.getMessage(`error_${state.error}`)),
      ];
    case 'clean':
    case 'ai':
      return [text('div', 'title', headline(state.verdict, threshold)), checkList(state.verdict, threshold)];
  }
}

function headline(verdict: OkVerdict, threshold: number): string {
  if (verdict.source !== 'model') return browser.i18n.getMessage('detailsLabeledAi');
  return browser.i18n.getMessage(isFlagged(verdict, threshold) ? 'detailsLikelyAi' : 'detailsClean');
}

function checkList(verdict: OkVerdict, threshold: number): HTMLElement {
  const message = (key: string, substitutions?: string[]) => browser.i18n.getMessage(key, substitutions);
  const metadata =
    verdict.source === 'metadata'
      ? row(message('badgeMetadata'), 'hit', message('detailsFound'), text('div', 'note', reasonText(verdict)))
      : statusRow(message('badgeMetadata'), verdict.metadata, message('detailsNoMetadata'));
  const provider = verdict.source === 'synthid' && (verdict.provider === 'openai' ? 'OpenAI' : 'Google');
  const watermark =
    verdict.source === 'model'
      ? statusRow(message('detailsWatermark'), verdict.watermark, message('detailsNoWatermark'))
      : verdict.source === 'watermark'
        ? row(
            message(VISIBLE_SCHEMES.has(verdict.scheme) ? 'detailsVisibleWatermark' : 'detailsInvisibleWatermark'),
            'hit',
            message('detailsWatermarkFound', [SCHEME_NAMES[verdict.scheme]]),
          )
      : provider
        ? row(message('detailsWatermark'), 'hit', message('detailsWatermarkFound', [provider]))
        : row(message('detailsWatermark'), 'off', message('detailsSkipped'));
  const models =
    verdict.source === 'model'
      ? row(
          message('detailsModels'),
          isFlagged(verdict, threshold) ? 'hit' : 'clear',
          percent(verdict.score),
          modelTable(verdict, threshold),
        )
      : row(message('detailsModels'), 'off', message('detailsSkipped'));
  const list = text('ul', 'checks');
  list.append(metadata, watermark, models);
  return list;
}

function statusRow(name: string, status: CheckStatus, clearText: string): HTMLElement {
  return status === 'clear' ? row(name, 'clear', clearText) : row(name, 'off', browser.i18n.getMessage('detailsOff'));
}

function row(name: string, status: 'hit' | 'clear' | 'off', value: string, extra?: HTMLElement): HTMLElement {
  const item = text('li', `check ${status}`);
  item.append(text('span', 'dot'), text('span', 'name', name), text('span', 'value', value));
  if (extra) item.append(extra);
  return item;
}

function modelTable(verdict: ModelVerdict, threshold: number): HTMLElement {
  const table = text('div', 'models');
  verdict.models.forEach((score, i) => {
    const bar = text('span', 'bar');
    bar.style.setProperty('--threshold', `${threshold * 100}%`);
    const fill = text('span');
    fill.style.width = `${score * 100}%`;
    bar.append(fill);
    table.append(text('span', 'name', model.models[i].label), bar, text('span', 'value', percent(score)));
  });
  const { weights } = model.calibration;
  const total = weights.reduce((sum, w) => sum + w, 0);
  const shares = weights.map((w) => Math.round((w / total) * 100)).join(' : ');
  const note =
    weights.length > 1
      ? browser.i18n.getMessage('detailsModelNote', [percent(threshold), shares])
      : browser.i18n.getMessage('detailsThreshold', [percent(threshold)]);
  table.append(text('div', 'note', note));
  return table;
}

function text(tag: string, className = '', content = ''): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content) node.textContent = content;
  return node;
}

let percentFormat: Intl.NumberFormat | null = null;

function percent(value: number): string {
  percentFormat ??= new Intl.NumberFormat(browser.i18n.getUILanguage(), { style: 'percent', maximumFractionDigits: 1 });
  return percentFormat.format(value);
}
