import type { ContentRequest, PageStats } from '../shared/messages';
import { isActiveOn, loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { Scanner } from './scanner';

let scanner: Scanner | null = null;
let active = false;
let reportTimer = 0;

function report(flagged: number, initial = false): void {
  clearTimeout(reportTimer);
  const send = () => void browser.runtime.sendMessage({ type: 'page-stats', flagged, initial }).catch(() => {});
  if (initial) send();
  else reportTimer = window.setTimeout(send, 250);
}

function apply(settings: Settings): void {
  active = isActiveOn(settings, location.hostname);
  if (active) {
    scanner ??= new Scanner(settings, (flagged) => report(flagged));
    scanner.restart(settings);
  } else {
    scanner?.stop();
  }
}

browser.runtime.onMessage.addListener((message: unknown) => {
  if ((message as ContentRequest)?.type !== 'get-stats' || window !== window.top) return undefined;
  const stats: PageStats = { ...(scanner?.stats ?? { flagged: 0, checked: 0 }), active };
  return Promise.resolve(stats);
});

if (document.documentElement instanceof HTMLHtmlElement) {
  report(0, true);
  void loadSettings().then(apply);
  onSettingsChanged(apply);
}
