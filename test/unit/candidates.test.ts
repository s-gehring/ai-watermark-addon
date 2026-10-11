import { describe, expect, it } from 'vitest';
import { firstCssUrl, isCheckableUrl, isLargeEnough } from '../../src/content/candidates';

describe('firstCssUrl', () => {
  it('extracts quoted, unquoted and escaped URLs', () => {
    expect(firstCssUrl('url("https://a.test/x.jpg")')).toBe('https://a.test/x.jpg');
    expect(firstCssUrl("url('/x.jpg')")).toBe('/x.jpg');
    expect(firstCssUrl('url(/x.jpg)')).toBe('/x.jpg');
    expect(firstCssUrl('url("/a\\"b.jpg")')).toBe('/a"b.jpg');
  });

  it('returns the first image of layered backgrounds and ignores gradients', () => {
    expect(firstCssUrl('linear-gradient(red, blue), url("/x.jpg"), url("/y.jpg")')).toBe('/x.jpg');
    expect(firstCssUrl('linear-gradient(red, blue)')).toBeNull();
  });
});

describe('isCheckableUrl', () => {
  it('accepts raster sources and rejects SVG and other schemes', () => {
    expect(isCheckableUrl('https://a.test/photo.jpg?w=800')).toBe(true);
    expect(isCheckableUrl('data:image/png;base64,AAAA')).toBe(true);
    expect(isCheckableUrl('blob:https://a.test/1234')).toBe(true);
    expect(isCheckableUrl('https://a.test/logo.svg')).toBe(false);
    expect(isCheckableUrl('https://a.test/logo.svg?v=2')).toBe(false);
    expect(isCheckableUrl('data:image/svg+xml;utf8,<svg/>')).toBe(false);
    expect(isCheckableUrl('chrome://branding/icon.png')).toBe(false);
  });
});

describe('isLargeEnough', () => {
  it('compares the shorter rendered side against the minimum', () => {
    expect(isLargeEnough({ width: 640, height: 200 }, 200)).toBe(true);
    expect(isLargeEnough({ width: 640, height: 199.5 }, 200)).toBe(false);
  });
});
