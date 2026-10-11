import { localize } from '../shared/i18n';
import {
  EMPTY_STATS,
  checkedCount,
  progressOf,
  sumStats,
  tabStatsKey,
  type FrameStats,
  type PageStatus,
} from '../shared/messages';
import { loadSettings, normalizeHost, saveSettings } from '../shared/settings';

const POLL_INTERVAL = 400;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const t = (key: string, ...substitutions: number[]) => browser.i18n.getMessage(key, substitutions.map(String));

async function main(): Promise<void> {
  localize();
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  const host = tab?.url && /^https?:/.test(tab.url) ? normalizeHost(new URL(tab.url).hostname) : null;
  const settings = await loadSettings();

  const enabled = $<HTMLInputElement>('enabled');
  const site = $<HTMLInputElement>('site');
  enabled.checked = settings.enabled;
  site.checked = host ? !settings.disabledHosts.includes(host) : false;
  site.disabled = !host || !settings.enabled;
  $('siteRow').hidden = !host;
  if (host) $('siteLabel').textContent = browser.i18n.getMessage('popupSiteEnabled', [host]);

  let status: PageStatus | null = null;
  const refreshStatus = async () => {
    status = tab?.id === undefined ? null : await browser.tabs
      .sendMessage(tab.id, { type: 'get-status' }, { frameId: 0 })
      .then((s) => s as PageStatus)
      .catch(() => null);
  };

  enabled.addEventListener('change', async () => {
    await saveSettings({ enabled: enabled.checked });
    site.disabled = !enabled.checked;
    setTimeout(refreshStatus, 200);
  });
  site.addEventListener('change', async () => {
    const current = await loadSettings();
    const others = current.disabledHosts.filter((h) => h !== host);
    await saveSettings({ disabledHosts: site.checked ? others : [...others, host!] });
    setTimeout(refreshStatus, 200);
  });
  $('options').addEventListener('click', () => {
    void browser.runtime.openOptionsPage();
    window.close();
  });

  await refreshStatus();
  const tick = async () => {
    render(tab?.id === undefined ? EMPTY_STATS : await readStats(tab.id), status);
    setTimeout(tick, POLL_INTERVAL);
  };
  await tick();
}

async function readStats(tabId: number): Promise<FrameStats> {
  const key = tabStatsKey(tabId);
  const stored = (await browser.storage.session.get(key))[key] as Record<string, FrameStats> | undefined;
  return sumStats(Object.values(stored ?? {}));
}

function render(stats: FrameStats, status: PageStatus | null): void {
  const checked = checkedCount(stats);
  const progress = progressOf(stats);
  $('flagged').textContent = String(stats.flagged);

  $('progress').hidden = progress === null;
  if (progress !== null) {
    const percent = Math.round(progress * 100);
    $('progressLabel').textContent = t('popupProgress', checked, checked + stats.pending);
    $('fill').style.width = `${percent}%`;
    $('bar').setAttribute('aria-valuenow', String(percent));
  }

  const parts: string[] = [];
  if (progress === null && checked > 0) parts.push(t('popupDone', checked));
  if (stats.clean > 0) parts.push(t('popupClean', stats.clean));
  if (stats.failed > 0) parts.push(t('popupFailed', stats.failed));
  if (checked === 0 && stats.pending === 0 && status?.active) parts.push(t('popupNone'));
  $('summary').textContent = parts.join(' · ');
  $('summary').hidden = parts.length === 0;

  const message = $('status');
  message.hidden = !!status?.active;
  message.textContent = t(status ? 'popupInactive' : 'popupUnavailable');
}

void main();
