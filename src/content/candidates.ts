const SVG_URL = /^data:image\/svg|\.svgz?(?:[?#]|$)/i;

export function firstCssUrl(value: string): string | null {
  const match = /url\(\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|([^)\s]*))\s*\)/i.exec(value);
  const raw = match?.[1] ?? match?.[2] ?? match?.[3];
  return raw ? raw.replace(/\\(.)/g, '$1') : null;
}

export function isCheckableUrl(url: string): boolean {
  if (SVG_URL.test(url)) return false;
  return /^(?:https?:|data:image\/|blob:)/i.test(url);
}

export function isLargeEnough(rect: { width: number; height: number }, minSize: number): boolean {
  return Math.min(rect.width, rect.height) >= minSize;
}

export function imageUrlOf(element: Element, includeBackgrounds: boolean): string | null {
  let url: string | null = null;
  if (element instanceof HTMLImageElement) {
    if (!element.complete || element.naturalWidth === 0) return null;
    url = element.currentSrc || element.src;
  } else if (includeBackgrounds) {
    const background = getComputedStyle(element).backgroundImage;
    url = background && background !== 'none' ? firstCssUrl(background) : null;
    if (url) url = new URL(url, document.baseURI).href;
  }
  return url && isCheckableUrl(url) ? url : null;
}

export function hasBackgroundImage(element: Element): boolean {
  const background = getComputedStyle(element).backgroundImage;
  return !!background && background !== 'none' && background.includes('url(');
}
