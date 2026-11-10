interface PageView {
  page: string;
  spent: number;
}

export class PageBudget {
  private readonly tabs = new Map<number, Map<number, PageView>>();

  constructor(private readonly limit: number) {}

  async load<T extends { bytes: ArrayBuffer }>(tabId: number, frameId: number, page: string, load: () => Promise<T | null>): Promise<T | null> {
    const view = this.viewOf(tabId, frameId, page);
    if (view.spent >= this.limit) return null;
    const loaded = await load();
    if (loaded) view.spent += loaded.bytes.byteLength;
    return loaded;
  }

  take(tabId: number, frameId: number, page: string): boolean {
    const view = this.viewOf(tabId, frameId, page);
    if (view.spent >= this.limit) return false;
    view.spent++;
    return true;
  }

  reset(tabId: number, frameId: number): void {
    this.tabs.get(tabId)?.delete(frameId);
  }

  forgetTab(tabId: number): void {
    this.tabs.delete(tabId);
  }

  private viewOf(tabId: number, frameId: number, page: string): PageView {
    let frames = this.tabs.get(tabId);
    if (!frames) this.tabs.set(tabId, (frames = new Map()));
    let view = frames.get(frameId);
    if (view?.page !== page) frames.set(frameId, (view = { page, spent: 0 }));
    return view;
  }
}
