import model from '../../model.json';

export interface Settings {
  enabled: boolean;
  minSize: number;
  threshold: number;
  checkMetadata: boolean;
  includeBackgrounds: boolean;
  showScore: boolean;
  disabledHosts: string[];
}

export const DEFAULT_SETTINGS: Readonly<Settings> = {
  enabled: true,
  minSize: 200,
  threshold: model.threshold,
  checkMetadata: true,
  includeBackgrounds: true,
  showScore: true,
  disabledHosts: [],
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export function normalizeSettings(raw: unknown): Settings {
  const input = (raw && typeof raw === 'object' ? raw : {}) as Partial<Record<keyof Settings, unknown>>;
  const bool = (key: 'enabled' | 'checkMetadata' | 'includeBackgrounds' | 'showScore') =>
    typeof input[key] === 'boolean' ? (input[key] as boolean) : DEFAULT_SETTINGS[key];
  const num = (key: 'minSize' | 'threshold', min: number, max: number) => {
    const value = Number(input[key]);
    return Number.isFinite(value) ? clamp(value, min, max) : DEFAULT_SETTINGS[key];
  };
  const hosts = Array.isArray(input.disabledHosts)
    ? [...new Set(input.disabledHosts.filter((h): h is string => typeof h === 'string').map(normalizeHost).filter(Boolean))]
    : [];
  return {
    enabled: bool('enabled'),
    minSize: Math.round(num('minSize', 16, 4096)),
    threshold: num('threshold', 0.01, 0.99),
    checkMetadata: bool('checkMetadata'),
    includeBackgrounds: bool('includeBackgrounds'),
    showScore: bool('showScore'),
    disabledHosts: hosts,
  };
}

export function normalizeHost(host: string): string {
  return host.trim().toLowerCase().replace(/^www\./, '');
}

export function isActiveOn(settings: Settings, hostname: string): boolean {
  return settings.enabled && !settings.disabledHosts.includes(normalizeHost(hostname));
}

export async function loadSettings(): Promise<Settings> {
  const { settings } = await browser.storage.local.get('settings');
  return normalizeSettings(settings);
}

export async function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const next = normalizeSettings({ ...(await loadSettings()), ...patch });
  await browser.storage.local.set({ settings: next });
  return next;
}

export function onSettingsChanged(listener: (settings: Settings) => void): void {
  browser.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes.settings) listener(normalizeSettings(changes.settings.newValue));
  });
}
