import { describe, expect, it } from 'vitest';
import { couldBeSourceOf } from '../../src/background/preprocess';
import {
  firstCssUrl,
  isCheckableUrl,
  isLargeEnough,
  knownOriginal,
  largestCandidate,
  parseSrcset,
} from '../../src/content/candidates';

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

describe('parseSrcset', () => {
  const base = 'https://a.test/page/';

  it('resolves URLs and reads width and density descriptors', () => {
    expect(parseSrcset('small.jpg 400w, /large.jpg 1600w', base)).toEqual([
      { url: 'https://a.test/page/small.jpg', width: 400 },
      { url: 'https://a.test/large.jpg', width: 1600 },
    ]);
    expect(parseSrcset('a.jpg, b.jpg 1.5x,c.jpg 2x', base)).toEqual([
      { url: 'https://a.test/page/a.jpg', density: 1 },
      { url: 'https://a.test/page/b.jpg', density: 1.5 },
      { url: 'https://a.test/page/c.jpg', density: 2 },
    ]);
  });

  it('keeps commas inside URLs', () => {
    expect(parseSrcset('https://cdn.test/w_400,c_limit/x.jpg 400w, https://cdn.test/w_800,c_limit/x.jpg 800w', base)).toEqual([
      { url: 'https://cdn.test/w_400,c_limit/x.jpg', width: 400 },
      { url: 'https://cdn.test/w_800,c_limit/x.jpg', width: 800 },
    ]);
  });
});

describe('largestCandidate', () => {
  const set = [
    { url: 'https://a.test/400.jpg', width: 400 },
    { url: 'https://a.test/1600.jpg', width: 1600 },
    { url: 'https://a.test/800.jpg', width: 800 },
  ];

  it('returns the widest candidate when it is clearly larger than the shown one', () => {
    expect(largestCandidate(set, 'https://a.test/400.jpg')).toBe('https://a.test/1600.jpg');
    expect(largestCandidate(set, 'https://a.test/1600.jpg')).toBeNull();
    expect(largestCandidate([...set, { url: 'https://a.test/1800.jpg', width: 1800 }], 'https://a.test/1600.jpg')).toBeNull();
  });

  it('treats the src outside a density set as 1x and ignores unknown current URLs in width sets', () => {
    expect(largestCandidate([{ url: 'https://a.test/2x.jpg', density: 2 }], 'https://a.test/1x.jpg')).toBe('https://a.test/2x.jpg');
    expect(largestCandidate(set, 'https://a.test/other.jpg')).toBeNull();
  });
});

describe('knownOriginal', () => {
  it('maps thumbnails of known image hosts to the original file', () => {
    expect(
      knownOriginal('https://thumb.wikimedia.org/wikipedia/commons/thumb/6/69/A%C3%A9.png/960px-A%C3%A9.png?utm_source=en.wikipedia.org'),
    ).toBe('https://upload.wikimedia.org/wikipedia/commons/6/69/A%C3%A9.png');
    expect(knownOriginal('https://upload.wikimedia.org/wikipedia/en/thumb/a/ab/X.jpg/250px-X.jpg')).toBe(
      'https://upload.wikimedia.org/wikipedia/en/a/ab/X.jpg',
    );
    expect(knownOriginal('https://preview.redd.it/abc123.png?width=640&crop=smart&auto=webp&s=f00')).toBe('https://i.redd.it/abc123.png');
    expect(knownOriginal('https://preview.redd.it/a-captive-audience-v0-hjdnilqr13wh1.jpeg?width=640&auto=webp&s=f00')).toBe(
      'https://i.redd.it/hjdnilqr13wh1.jpeg',
    );
    expect(knownOriginal('https://blog.test/wp-content/uploads/2026/05/photo-1024x683.jpg')).toBe(
      'https://blog.test/wp-content/uploads/2026/05/photo.jpg',
    );
    expect(knownOriginal('https://blog.test/wp-content/uploads/2026/05/photo-scaled.jpg')).toBe(
      'https://blog.test/wp-content/uploads/2026/05/photo.jpg',
    );
    expect(knownOriginal('https://news.test/wp-content/uploads/2026/05/photo.jpg?quality=90&w=680')).toBe(
      'https://news.test/wp-content/uploads/2026/05/photo.jpg',
    );
    expect(knownOriginal('https://lh3.googleusercontent.com/AbC-dEf_123=w800-h600-rw')).toBe('https://lh3.googleusercontent.com/AbC-dEf_123=s0');
  });

  it('leaves originals, vector thumbnails and unknown hosts alone', () => {
    expect(knownOriginal('https://upload.wikimedia.org/wikipedia/commons/6/69/X.png')).toBeNull();
    expect(knownOriginal('https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Sign.svg/500px-Sign.svg.png')).toBeNull();
    expect(knownOriginal('https://blog.test/wp-content/uploads/2026/05/photo.jpg')).toBeNull();
    expect(knownOriginal('https://blog.test/wp-content/uploads/2026/05/photo.jpg?ver=2')).toBeNull();
    expect(knownOriginal('https://lh3.googleusercontent.com/AbC-dEf_123=s0')).toBeNull();
    expect(knownOriginal('https://cdn.test/photo-1024x683.jpg')).toBeNull();
  });
});

describe('couldBeSourceOf', () => {
  it('accepts an image of the same aspect ratio that is at least as large', () => {
    expect(couldBeSourceOf({ width: 2048, height: 1365 }, { width: 330, height: 220 })).toBe(true);
    expect(couldBeSourceOf({ width: 330, height: 220 }, { width: 330, height: 220 })).toBe(true);
  });

  it('rejects smaller images and other crops', () => {
    expect(couldBeSourceOf({ width: 300, height: 200 }, { width: 330, height: 220 })).toBe(false);
    expect(couldBeSourceOf({ width: 2048, height: 2048 }, { width: 330, height: 220 })).toBe(false);
  });
});
