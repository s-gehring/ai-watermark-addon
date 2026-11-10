import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  commonsFilePage,
  deviantArtLabels,
  hasAiCategory,
  isAiCategory,
  isLookupKey,
  labelKeyOf,
  lookupBatches,
  lookupUrl,
  parseLookup,
  platformOf,
} from '../../src/shared/platform-labels';

const fixture = (name: string) => readFileSync(new URL(`./fixtures/platform-labels/${name}`, import.meta.url), 'utf8');
const inlineScript = (html: string) => /<script>([\s\S]*)<\/script>/.exec(html)![1];

describe('platformOf', () => {
  it('knows the pages of pixiv, DeviantArt and the Wikimedia projects', () => {
    expect(platformOf('www.pixiv.net')).toBe('pixiv');
    expect(platformOf('www.deviantart.com')).toBe('deviantart');
    expect(platformOf('commons.wikimedia.org')).toBe('commons');
    expect(platformOf('de.wikipedia.org')).toBe('commons');
    expect(platformOf('en.m.wikipedia.org')).toBe('commons');
  });

  it('leaves every other site alone', () => {
    for (const host of ['example.com', 'notpixiv.net', 'sketch.pixiv.net', 'wikipedia.org.example.com', 'civitai.com']) {
      expect(platformOf(host)).toBeNull();
    }
  });
});

describe('pixiv', () => {
  const response = JSON.parse(fixture('pixiv-illusts.json'));
  const urls: string[] = response.body.illusts.map((work: { url: string }) => work.url);

  it('reads the work id from thumbnails, custom thumbnails and hashed file names', () => {
    expect(urls.map((url) => labelKeyOf('pixiv', url))).toEqual(['136500000', '137590173', '130000000', '125000000']);
    expect(labelKeyOf('pixiv', 'https://i.pximg.net/img-original/img/2025/10/20/17/13/22/136500000_p0.png')).toBe('136500000');
  });

  it('finds no work behind avatars, novel covers and other hosts', () => {
    expect(labelKeyOf('pixiv', 'https://i.pximg.net/user-profile/img/2024/03/01/10/20/30/25600000_0123456789abcdef_170.jpg')).toBeNull();
    expect(labelKeyOf('pixiv', 'https://i.pximg.net/c/600x600/novel-cover-master/img/2025/01/02/03/04/05/23456789_ab12_master1200.jpg')).toBeNull();
    expect(labelKeyOf('pixiv', 'https://s.pximg.net/common/images/ogp.png')).toBeNull();
    expect(labelKeyOf('pixiv', 'https://example.com/img-master/img/2025/10/20/17/13/22/136500000_p0_master1200.jpg')).toBeNull();
  });

  it('counts only aiType 2 as labeled and answers only for the works in the response', () => {
    const { labels, truncated } = parseLookup('pixiv', response);
    expect([...labels]).toEqual([
      ['136500000', true],
      ['137590173', true],
      ['130000000', true],
      ['125000000', false],
    ]);
    expect(labels.has('151848495')).toBe(false);
    expect(truncated).toBe(false);
  });

  it('asks for several works in one request to pixiv', () => {
    expect(lookupUrl('pixiv', ['1', '22'])).toBe('https://www.pixiv.net/ajax/illust/recommend/illusts?illust_ids%5B%5D=1&illust_ids%5B%5D=22');
  });

  it('survives an error response', () => {
    expect(parseLookup('pixiv', { error: true, message: 'x', body: [] }).labels.size).toBe(0);
    expect(parseLookup('pixiv', null).labels.size).toBe(0);
  });
});

describe('DeviantArt', () => {
  const dreamup = deviantArtLabels(inlineScript(fixture('deviantart-tag-dreamup.html')));
  const landscape = deviantArtLabels(inlineScript(fixture('deviantart-tag-landscape.html')));

  it('reads isAiGenerated per deviation from the embedded page state', () => {
    expect([...dreamup.values()]).toEqual([true, true, true]);
    expect([...landscape.values()]).toEqual([false, false, false]);
  });

  it('matches the displayed renditions of a deviation and nothing else', () => {
    const [file] = dreamup.keys();
    expect(file).toBe(
      'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/2da1c90a-37f6-48a9-b3f6-4e05f9a8273e/dfi325b-359f7dc4-83b4-4e4b-b642-c00659e1fc14.jpg',
    );
    expect(labelKeyOf('deviantart', `${file}/v1/fill/w_300,h_200,q_70,strp/some_title_by_someone_dfi325b-300w.jpg?token=abc.def`)).toBe(file);
    expect(labelKeyOf('deviantart', `${file}?token=abc.def`)).toBe(file);
    expect(labelKeyOf('deviantart', file.replace('.com/f/', '.com/intermediary/f/') + '/v1/fit/w_414,h_240,q_70,strp/x.jpg')).toBe(file);
    expect(labelKeyOf('deviantart', 'https://a.deviantart.net/avatars-big/s/o/someone.jpg?8')).toBeNull();
    expect(labelKeyOf('deviantart', 'https://st.deviantart.net/badges/glass.png?5')).toBeNull();
  });

  it('returns nothing for scripts without a readable state', () => {
    expect(deviantArtLabels('window.__OTHER__ = JSON.parse("{}");').size).toBe(0);
    expect(deviantArtLabels('window.__INITIAL_STATE__ = JSON.parse("{\\"@@entities\\":').size).toBe(0);
    expect(deviantArtLabels('window.__INITIAL_STATE__ = JSON.parse("not json");').size).toBe(0);
  });
});

describe('Wikimedia Commons', () => {
  const response = JSON.parse(fixture('commons-categories.json'));

  it('reads the file title from thumbnails and originals on Commons only', () => {
    expect(
      labelKeyOf(
        'commons',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Home_Caim%C3%A1n_-_IA_Midjourney_versi%C3%B3n_2.png/330px-Home_Caim%C3%A1n_-_IA_Midjourney_versi%C3%B3n_2.png',
      ),
    ).toBe('File:Home Caimán - IA Midjourney versión 2.png');
    expect(labelKeyOf('commons', 'https://upload.wikimedia.org/wikipedia/commons/5/5c/Vaporwave_sunset.jpg')).toBe('File:Vaporwave sunset.jpg');
    expect(labelKeyOf('commons', 'https://thumb.wikimedia.org/wikipedia/commons/thumb/5/5c/Vaporwave_sunset.jpg/960px-Vaporwave_sunset.jpg')).toBe(
      'File:Vaporwave sunset.jpg',
    );
    expect(labelKeyOf('commons', 'https://upload.wikimedia.org/wikipedia/en/thumb/a/a9/Example.jpg/250px-Example.jpg')).toBeNull();
    expect(labelKeyOf('commons', 'https://upload.wikimedia.org/wikipedia/commons/thumb/archive/a/a9/2020%21Example.jpg/120px-Example.jpg')).toBeNull();
    expect(labelKeyOf('commons', 'https://upload.wikimedia.org/wikipedia/commons/a/a9/%E0%A4%A.jpg')).toBeNull();
    expect(labelKeyOf('commons', 'https://commons.wikimedia.org/static/images/project-logos/commonswiki.png')).toBeNull();
  });

  it('labels files in categories for AI-generated images', () => {
    const { labels, truncated } = parseLookup('commons', response);
    expect(labels.get('File:Home Caimán - IA Midjourney versión 2.png')).toBe(true);
    expect(labels.get('File:Vaporwave sunset.jpg')).toBe(true);
    expect(labels.get('File:Vaporwave_sunset.jpg')).toBe(true);
    expect(truncated).toBe(false);
  });

  it('does not label photographs of AI images, AI restorations, other files and missing files', () => {
    const { labels } = parseLookup('commons', response);
    expect(labels.get('File:Wikimedia Czech Republic Stickers - Máte tam chybu.jpg')).toBe(false);
    expect(labels.get('File:Chopin 1849 by Bisson (AI restoration).jpg')).toBe(false);
    expect(labels.get('File:Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg')).toBe(false);
    expect(labels.get('File:Does not exist 12345.jpg')).toBe(false);
  });

  it('answers only for files whose category list is complete when Commons cuts the response off', () => {
    const { labels, truncated } = parseLookup('commons', JSON.parse(fixture('commons-categories-truncated.json')));
    expect(truncated).toBe(true);
    expect([...labels]).toEqual([['File:Does not exist 12345.jpg', false]]);
    const cutLater = { ...response, continue: { clcontinue: '154241961|Vaporwave', continue: '||' } };
    expect([...parseLookup('commons', cutLater).labels.keys()]).toEqual([
      'File:Does not exist 12345.jpg',
      'File:Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg',
      'File:Home Caimán - IA Midjourney versión 2.png',
      'File:Wikimedia Czech Republic Stickers - Máte tam chybu.jpg',
    ]);
  });

  it('keeps the request URL short for long non-Latin titles', () => {
    const long = Array.from({ length: 20 }, (_, i) => `File:${'画'.repeat(200)}${i}.jpg`);
    const batches = lookupBatches('commons', long);
    expect(batches.flat()).toEqual(long);
    expect(batches.length).toBeGreaterThan(1);
    for (const batch of batches) expect(lookupUrl('commons', batch).length).toBeLessThan(4300);
    const short = Array.from({ length: 45 }, (_, i) => `File:Short ${i}.jpg`);
    expect(lookupBatches('commons', short).map((batch) => batch.length)).toEqual([20, 20, 5]);
    expect(lookupBatches('pixiv', Array.from({ length: 120 }, (_, i) => String(i))).map((batch) => batch.length)).toEqual([50, 50, 20]);
  });

  it('tells categories of AI images from categories about them', () => {
    for (const name of ['AI-generated images', 'Images generated by Midjourney', 'Unidentified AI-generated images', 'Witches in AI generated art']) {
      expect(isAiCategory(name)).toBe(true);
    }
    for (const name of ['PD-algorithm', 'AI-assisted images', 'Videos about AI-generated images', 'Unconfirmed likely AI-generated images to check', 'Self-published work']) {
      expect(isAiCategory(name)).toBe(false);
    }
    expect(hasAiCategory(['CC-Zero', 'Images generated by DALL-E'])).toBe(true);
    expect(hasAiCategory(['AI-generated cats', 'Photographs of AI-generated images'])).toBe(false);
  });

  it('asks Commons anonymously for the categories of several files', () => {
    const url = new URL(lookupUrl('commons', ['File:A b.jpg', 'File:Ü&1.png']));
    expect(url.origin + url.pathname).toBe('https://commons.wikimedia.org/w/api.php');
    expect(Object.fromEntries(url.searchParams)).toEqual({
      action: 'query',
      format: 'json',
      formatversion: '2',
      origin: '*',
      prop: 'categories',
      cllimit: 'max',
      titles: 'File:A b.jpg|File:Ü&1.png',
    });
  });

  it('recognizes file pages on Commons', () => {
    expect(commonsFilePage('commons.wikimedia.org', '/wiki/File:Vaporwave_sunset.jpg')).toBe('File:Vaporwave sunset.jpg');
    expect(commonsFilePage('commons.wikimedia.org', '/wiki/Category:AI-generated_images')).toBeNull();
    expect(commonsFilePage('en.wikipedia.org', '/wiki/File:Vaporwave_sunset.jpg')).toBeNull();
  });
});

describe('isLookupKey', () => {
  it('accepts only work ids and file titles', () => {
    expect(isLookupKey('pixiv', '136500000')).toBe(true);
    expect(isLookupKey('pixiv', '13&x=1')).toBe(false);
    expect(isLookupKey('commons', 'File:Vaporwave sunset.jpg')).toBe(true);
    expect(isLookupKey('commons', 'File:A.jpg|File:B.jpg')).toBe(false);
    expect(isLookupKey('commons', 'Category:AI-generated images')).toBe(false);
  });
});
