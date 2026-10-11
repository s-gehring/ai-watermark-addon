import { describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS, isActiveOn, normalizeSettings } from '../../src/shared/settings';

describe('normalizeSettings', () => {
  it('falls back to defaults for missing or invalid input', () => {
    expect(normalizeSettings(undefined)).toEqual(DEFAULT_SETTINGS);
    expect(normalizeSettings({ minSize: 'big', enabled: 'yes' })).toEqual(DEFAULT_SETTINGS);
  });

  it('clamps numbers and normalizes hosts', () => {
    const settings = normalizeSettings({
      minSize: 5,
      threshold: 1.5,
      disabledHosts: [' WWW.Example.com ', 'example.com', 42, ''],
    });
    expect(settings.minSize).toBe(16);
    expect(settings.threshold).toBe(0.99);
    expect(settings.disabledHosts).toEqual(['example.com']);
  });
});

describe('isActiveOn', () => {
  it('respects the global switch and disabled sites', () => {
    const settings = normalizeSettings({ disabledHosts: ['example.com'] });
    expect(isActiveOn(settings, 'www.example.com')).toBe(false);
    expect(isActiveOn(settings, 'example.org')).toBe(true);
    expect(isActiveOn({ ...settings, enabled: false }, 'example.org')).toBe(false);
  });
});
