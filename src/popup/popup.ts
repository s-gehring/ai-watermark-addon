import { localize } from '../shared/i18n';
import type { PageStats } from '../shared/messages';
import { loadSettings, normalizeHost, saveSettings } from '../shared/settings';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

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

  enabled.addEventListener('change', async () => {
    await saveSettings({ enabled: enabled.checked });
    site.disabled = !enabled.checked;
    void refreshStats(tab?.id, 200);
  });
  site.addEventListener('change', async () => {
    const current = await loadSettings();
    const others = current.disabledHosts.filter((h) => h !== host);
    await saveSettings({ disabledHosts: site.checked ? others : [...others, host!] });
    void refreshStats(tab?.id, 200);
  });
  $('options').addEventListener('click', () => {
    void browser.runtime.openOptionsPage();
    window.close();
  });

  await refreshStats(tab?.id);
}

async function refreshStats(tabId: number | undefined, delay = 0): Promise<void> {
  const status = $('status');
  let stats: PageStats | null = null;
  if (tabId !== undefined) {
    await new Promise((resolve) => setTimeout(resolve, delay));
    stats = await browser.tabs
      .sendMessage(tabId, { type: 'get-stats' }, { frameId: 0 })
      .then((s) => s as PageStats)
      .catch(() => null);
  }
  const flagged = tabId !== undefined ? await browser.action.getBadgeText({ tabId }).catch(() => '') : '';
  $('flagged').textContent = flagged || '0';
  $('checked').textContent = String(stats?.checked ?? 0);
  status.hidden = !!stats?.active;
  status.textContent = browser.i18n.getMessage(stats ? 'popupInactive' : 'popupUnavailable');
}

void main();
