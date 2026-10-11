import { EMPTY_STATS, type ContentRequest, type FrameStats, type PageStatus } from '../shared/messages';
import { isActiveOn, loadSettings, onSettingsChanged, type Settings } from '../shared/settings';
import { Scanner } from './scanner';

const REPORT_INTERVAL = 250;

let scanner: Scanner | null = null;
let active = false;
let latest: FrameStats = { ...EMPTY_STATS };
let reportTimer = 0;

function send(initial: boolean): void {
  void browser.runtime.sendMessage({ type: 'page-stats', stats: latest, initial }).catch(() => {});
}

function report(stats: FrameStats): void {
  latest = stats;
  if (reportTimer) return;
  reportTimer = window.setTimeout(() => {
    reportTimer = 0;
    send(false);
  }, REPORT_INTERVAL);
}

function apply(settings: Settings): void {
  active = isActiveOn(settings, location.hostname);
  if (active) {
    scanner ??= new Scanner(settings, report);
    scanner.restart(settings);
  } else {
    scanner?.stop();
  }
}

browser.runtime.onMessage.addListener((message: unknown) => {
  if ((message as ContentRequest)?.type !== 'get-status' || window !== window.top) return undefined;
  const status: PageStatus = { active };
  return Promise.resolve(status);
});

if (document.documentElement instanceof HTMLHtmlElement) {
  send(true);
  void loadSettings().then(apply);
  onSettingsChanged(apply);
}
