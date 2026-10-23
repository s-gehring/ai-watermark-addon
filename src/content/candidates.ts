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

const MIN_GAIN = 1.25;
const RASTER_PATH = /\.(?:jpe?g|png|webp|avif|gif)$/i;

const ORIGINAL_RULES: [RegExp, string][] = [
  [/^https:\/\/(?:upload|thumb)\.wikimedia\.org\/([\w-]+\/[\w-]+)\/thumb\/(\w\/\w\w\/[^/?#]+\.(?:jpe?g|png|webp|gif))\/[^/?#]+(?:[?#].*)?$/i, 'https://upload.wikimedia.org/$1/$2'],
  [/^https:\/\/preview\.redd\.it\/(?:[^/?#]*-v\d+-)?([a-z0-9]+\.(?:jpe?g|png|webp|gif))(?:[?#].*)?$/i, 'https://i.redd.it/$1'],
  [/^(https?:\/\/[^?#]+\/wp-content\/uploads\/[^?#]+?)(?:-scaled|-\d+x\d+)(\.(?:jpe?g|png|webp|avif|gif))(?:\?[^#]*)?$/i, '$1$2'],
  [/^(https?:\/\/[^?#]+\/wp-content\/uploads\/[^?#]+\.(?:jpe?g|png|webp|avif|gif))\?(?:[^#]*&)?(?:w|h|resize|fit)=[^#]*$/i, '$1'],
  [/^(https:\/\/[\w-]+\.googleusercontent\.com\/[^?#=]+)=[^/?#]+$/i, '$1=s0'],
];

export interface SrcsetCandidate {
  url: string;
  width?: number;
  density?: number;
}

export function parseSrcset(srcset: string, base: string): SrcsetCandidate[] {
  const candidates: SrcsetCandidate[] = [];
  let i = 0;
  while (i < srcset.length) {
    while (i < srcset.length && /[\s,]/.test(srcset[i])) i++;
    const start = i;
    while (i < srcset.length && !/\s/.test(srcset[i])) i++;
    let raw = srcset.slice(start, i);
    let descriptors: string[] = [];
    if (raw.endsWith(',')) {
      raw = raw.replace(/,+$/, '');
    } else {
      const end = srcset.indexOf(',', i);
      descriptors = srcset.slice(i, end < 0 ? srcset.length : end).trim().split(/\s+/);
      i = end < 0 ? srcset.length : end + 1;
    }
    if (!raw) continue;
    let url: string;
    try {
      url = new URL(raw, base).href;
    } catch {
      continue;
    }
    const width = descriptors.find((d) => /^\d+w$/.test(d));
    const density = descriptors.find((d) => /^\d*\.?\d+x$/.test(d));
    if (width) candidates.push({ url, width: parseInt(width, 10) });
    else candidates.push({ url, density: density ? parseFloat(density) : 1 });
  }
  return candidates;
}

export function largestCandidate(candidates: SrcsetCandidate[], current: string): string | null {
  const selected =
    candidates.find((c) => c.url === current) ??
    (candidates.every((c) => c.width === undefined) ? { url: current, density: 1 } : undefined);
  if (!selected) return null;
  const size = (c: SrcsetCandidate) => (selected.width !== undefined ? c.width : c.density) ?? 0;
  const largest = candidates.reduce((best, c) => (size(c) > size(best) ? c : best), selected);
  return size(largest) >= size(selected) * MIN_GAIN ? largest.url : null;
}

const WAGTAIL_RENDITION =
  /^(https:\/\/storage\.googleapis\.com\/gweb-[\w-]+)\/images\/([^/?#]+?)(?:\.[0-9a-f]{8})?\.(?:width|height|fill|max|min|scale)-[^/?#]*\.(png|jpe?g|gif|webp)(?:[?#].*)?$/i;

export function knownOriginals(url: string): string[] {
  const rendition = WAGTAIL_RENDITION.exec(url);
  if (rendition) {
    const [, bucket, name, extension] = rendition;
    const extensions = extension.toLowerCase() === 'webp' ? ['png', 'jpg', 'jpeg'] : [extension];
    return extensions.map((e) => `${bucket}/original_images/${name}.${e}`);
  }
  for (const [pattern, replacement] of ORIGINAL_RULES) {
    if (!pattern.test(url)) continue;
    const original = url.replace(pattern, replacement);
    return original !== url ? [original] : [];
  }
  return [];
}

function linkedImage(image: HTMLImageElement): string | null {
  const link = image.closest('a[href]');
  if (!(link instanceof HTMLAnchorElement) || !/^https?:$/.test(link.protocol)) return null;
  return RASTER_PATH.test(link.pathname) ? link.href : null;
}

function srcsetCandidate(image: HTMLImageElement, current: string): string | null {
  const picture = image.parentElement instanceof HTMLPictureElement ? image.parentElement : null;
  const sources = picture ? [...picture.querySelectorAll(':scope > source[srcset]')] : [];
  for (const element of [...sources, image]) {
    const candidates = parseSrcset(element.getAttribute('srcset') ?? '', document.baseURI);
    if (element !== image && !candidates.some((c) => c.url === current)) continue;
    return largestCandidate(candidates, current);
  }
  return null;
}

export function largerVariantsOf(image: HTMLImageElement, current: string): string[] {
  if (!/^https?:/.test(current)) return [];
  const known = knownOriginals(current);
  if (known.length) return known;
  const origins = [new URL(current).origin, location.origin];
  const allowed = (url: string | null): url is string => !!url && url !== current && origins.includes(new URL(url).origin);
  const linked = linkedImage(image);
  if (allowed(linked)) return [linked];
  const largest = srcsetCandidate(image, current);
  return allowed(largest) ? [largest] : [];
}

export function hasBackgroundImage(element: Element): boolean {
  const background = getComputedStyle(element).backgroundImage;
  return !!background && background !== 'none' && background.includes('url(');
}
