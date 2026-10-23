export const LAYER_TAG = 'ai-watermark-layer';

export function shadowRootOf(element: Element): ShadowRoot | null {
  if (element.localName === LAYER_TAG) return null;
  return (element as Element & { openOrClosedShadowRoot?: ShadowRoot | null }).openOrClosedShadowRoot ?? element.shadowRoot;
}

export function parentOf(element: Element): Element | null {
  if (element.parentElement) return element.parentElement;
  const root = element.getRootNode();
  return root instanceof ShadowRoot ? root.host : null;
}

export function composedContains(ancestor: Element, element: Element): boolean {
  for (let node: Element | null = element; node; node = parentOf(node)) {
    if (node === ancestor) return true;
  }
  return false;
}
