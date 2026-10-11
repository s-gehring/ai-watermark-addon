import { describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS, fromStorage, isActiveOn, normalizeSettings, toStorage } from '../../src/shared/settings';

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

describe('fromStorage', () => {
  it('keeps a threshold saved under the current calibration', () => {
    const saved = toStorage({ ...DEFAULT_SETTINGS, threshold: 0.5, minSize: 300 });
    expect(fromStorage(saved)).toMatchObject({ threshold: 0.5, minSize: 300 });
  });

  it('resets a threshold saved under another calibration and keeps the rest', () => {
    const legacy = { ...DEFAULT_SETTINGS, threshold: 0.5, minSize: 300 };
    expect(fromStorage(legacy)).toMatchObject({ threshold: DEFAULT_SETTINGS.threshold, minSize: 300 });
    expect(fromStorage({ ...toStorage(legacy), calibration: 'other' }).threshold).toBe(DEFAULT_SETTINGS.threshold);
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
