import { progressOf, sumStats, tabStatsKey, type FrameStats } from '../shared/messages';
import { progressIcon } from './action-icon';

const BADGE_COLOR = '#6d28d9';
const DEFAULT_ICON = 'icons/icon.svg';

type Frames = Record<string, FrameStats>;

const frames = new Map<number, Promise<Frames>>();
const shownIcon = new Map<number, string>();

function framesOf(tabId: number): Promise<Frames> {
  let loaded = frames.get(tabId);
  if (!loaded) {
    const key = tabStatsKey(tabId);
    loaded = browser.storage.session.get(key).then((stored) => (stored[key] as Frames | undefined) ?? {});
    frames.set(tabId, loaded);
  }
  return loaded;
}

export async function updateTabStats(tabId: number, frameId: number, stats: FrameStats, initial: boolean): Promise<void> {
  if (initial && frameId === 0) {
    frames.set(tabId, Promise.resolve({}));
    shownIcon.delete(tabId);
  }
  const current = await framesOf(tabId);
  current[frameId] = stats;
  await browser.storage.session.set({ [tabStatsKey(tabId)]: current });

  const total = sumStats(Object.values(current));
  await browser.action.setBadgeBackgroundColor({ tabId, color: BADGE_COLOR });
  await browser.action.setBadgeText({ tabId, text: total.flagged > 0 ? String(total.flagged) : '' });

  const progress = progressOf(total);
  const icon = progress === null ? null : progressIcon(progress);
  const key = icon?.key ?? 'default';
  if (shownIcon.get(tabId) === key) return;
  shownIcon.set(tabId, key);
  await (icon
    ? browser.action.setIcon({ tabId, imageData: icon.imageData })
    : browser.action.setIcon({ tabId, path: DEFAULT_ICON }));
}

export function forgetTab(tabId: number): void {
  frames.delete(tabId);
  shownIcon.delete(tabId);
  void browser.storage.session.remove(tabStatsKey(tabId));
}
