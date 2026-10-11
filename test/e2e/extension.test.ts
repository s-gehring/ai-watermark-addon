import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { fileURLToPath } from 'node:url';
import { download } from 'geckodriver';
import sharp from 'sharp';
import { Builder, until } from 'selenium-webdriver';
import firefox from 'selenium-webdriver/firefox.js';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

const ADDON_ID = 'ai-watermark-addon@s-gehring.github.io';
const ADDON_UUID = '6f0c3a52-9a51-4c1e-8f7e-2f6f5d1b7a10';
const DIST = fileURLToPath(new URL('../../dist', import.meta.url));

const XMP_AI = `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/></rdf:RDF></x:xmpmeta>`;

async function photoLikeJpeg(): Promise<Buffer> {
  const width = 480;
  const height = 360;
  const pixels = Buffer.alloc(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      pixels[i] = (x * 255) / width;
      pixels[i + 1] = (y * 255) / height;
      pixels[i + 2] = ((x ^ y) & 0xff) / 2;
    }
  }
  return sharp(pixels, { raw: { width, height, channels: 3 } }).jpeg({ quality: 85 }).toBuffer();
}

function withXmp(jpeg: Buffer, xmp: string): Buffer {
  const payload = Buffer.concat([Buffer.from('http://ns.adobe.com/xap/1.0/\x00', 'latin1'), Buffer.from(xmp, 'utf8')]);
  const header = Buffer.from([0xff, 0xe1, (payload.length + 2) >> 8, (payload.length + 2) & 0xff]);
  return Buffer.concat([jpeg.subarray(0, 2), header, payload, jpeg.subarray(2)]);
}

const PAGE = `<!doctype html>
<html><head><meta charset="utf-8"><style>
  body { margin: 0; font: 14px sans-serif; }
  img { display: block; }
  .box { width: 320px; height: 240px; margin: 8px; }
  #bg { background-image: url("/ai.jpg?bg"); background-size: cover; }
  #far { margin-top: 3000px; }
</style></head><body>
  <img id="meta" class="box" src="/ai.jpg">
  <div id="bg" class="box"></div>
  <img id="small" src="/ai.jpg?small" width="100" height="100">
  <img id="plain" class="box" src="/plain.jpg">
  <img id="far" class="box" src="/ai.jpg?far">
  <script>
    setTimeout(() => {
      const img = document.createElement('img');
      img.id = 'late';
      img.className = 'box';
      img.src = '/ai.jpg?late';
      document.body.insertBefore(img, document.getElementById('small'));
    }, 800);
  </script>
</body></html>`;

let server: Server;
let origin: string;
let driver: firefox.Driver;

beforeAll(async () => {
  const plain = await photoLikeJpeg();
  const ai = withXmp(plain, XMP_AI);
  server = createServer((req, res) => {
    const path = new URL(req.url!, 'http://x').pathname;
    if (path === '/') res.writeHead(200, { 'content-type': 'text/html' }).end(PAGE);
    else if (path === '/ai.jpg') res.writeHead(200, { 'content-type': 'image/jpeg' }).end(ai);
    else if (path === '/plain.jpg') res.writeHead(200, { 'content-type': 'image/jpeg' }).end(plain);
    else res.writeHead(404).end();
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

  const options = new firefox.Options()
    .addArguments('-headless')
    .windowSize({ width: 1200, height: 1400 })
    .setPreference('extensions.webextensions.uuids', JSON.stringify({ [ADDON_ID]: ADDON_UUID }));
  driver = (await new Builder()
    .forBrowser('firefox')
    .setFirefoxOptions(options)
    .setFirefoxService(new firefox.ServiceBuilder(await download()).addArguments('--allow-system-access'))
    .build()) as firefox.Driver;
  await driver.installAddon(DIST, true);
}, 120_000);

afterAll(async () => {
  await driver?.quit();
  server?.close();
});

interface BadgeInfo {
  kind: string;
  text: string;
  owner: string | null;
}

interface TabStats {
  flagged: number;
  clean: number;
  pending: number;
  failed: number;
  badgeText: string;
}

async function openExtensionPage(path: string): Promise<void> {
  await driver.setContext(firefox.Context.CHROME);
  await driver.executeScript(
    (url: string) => {
      const g = globalThis as unknown as {
        Services: {
          wm: { getMostRecentWindow(type: string): { gBrowser: { selectedTab: unknown; addTab(url: string, options: object): unknown } } };
          scriptSecurityManager: { getSystemPrincipal(): unknown };
        };
      };
      const { gBrowser } = g.Services.wm.getMostRecentWindow('navigator:browser');
      gBrowser.selectedTab = gBrowser.addTab(url, { triggeringPrincipal: g.Services.scriptSecurityManager.getSystemPrincipal() });
    },
    `moz-extension://${ADDON_UUID}/${path}`,
  );
  await driver.setContext(firefox.Context.CONTENT);
  await driver.wait(async () => (await driver.getAllWindowHandles()).length > 1, 10_000);
  await driver.switchTo().window((await driver.getAllWindowHandles()).at(-1)!);
}

async function visibleBadges(): Promise<BadgeInfo[]> {
  return driver.executeScript<BadgeInfo[]>(() => {
    const layer = document.querySelector('ai-watermark-layer');
    const badges = [...(layer?.shadowRoot?.querySelectorAll<HTMLElement>('.badge:not([hidden])') ?? [])];
    return badges.map((badge) => {
      const b = badge.getBoundingClientRect();
      const owner = [...document.querySelectorAll<HTMLElement>('[id]')].find((el) => {
        const r = el.getBoundingClientRect();
        return b.left >= r.left && b.right <= r.right && b.top >= r.top && b.bottom <= r.bottom;
      });
      const kind = ['ai', 'clean', 'pending', 'failed'].find((k) => badge.classList.contains(k)) ?? '';
      return { kind, text: badge.textContent ?? '', owner: owner?.id ?? null };
    });
  });
}

async function pendingCount(): Promise<number> {
  return driver.executeScript<number>(
    () => document.querySelector('ai-watermark-layer')?.shadowRoot?.querySelectorAll('.badge.pending').length ?? 0,
  );
}

async function waitForBadges(expected: Record<string, string[]>): Promise<BadgeInfo[]> {
  let last: BadgeInfo[] = [];
  await driver.wait(async () => {
    last = await visibleBadges();
    const matched = Object.entries(expected).every(([owner, kinds]) =>
      last.some((b) => b.owner === owner && kinds.includes(b.kind)),
    );
    return matched && (await pendingCount()) === 0;
  }, 30_000).catch(() => {});
  return last;
}

const kindOf = (badges: BadgeInfo[], owner: string) => badges.find((b) => b.owner === owner)?.kind;

describe('content script', () => {
  it('marks images with AI metadata, including CSS backgrounds and late additions', async () => {
    await driver.get(`${origin}/`);
    const badges = await waitForBadges({ meta: ['ai'], bg: ['ai'], late: ['ai'], plain: ['ai', 'clean'] });
    expect(kindOf(badges, 'meta')).toBe('ai');
    expect(kindOf(badges, 'bg')).toBe('ai');
    expect(kindOf(badges, 'late')).toBe('ai');
    expect(badges.find((b) => b.owner === 'meta')?.text).toMatch(/^AI/);
    expect(['ai', 'clean']).toContain(kindOf(badges, 'plain'));
    expect(kindOf(badges, 'small')).toBeUndefined();
  }, 60_000);

  it('checks images that scroll into view', async () => {
    await driver.executeScript(() => document.getElementById('far')!.scrollIntoView());
    const badges = await waitForBadges({ far: ['ai'] });
    expect(kindOf(badges, 'far')).toBe('ai');
  }, 60_000);
});

describe('background', () => {
  it('aggregates the page progress for the popup and the toolbar badge', async () => {
    const pageHandle = await driver.getWindowHandle();
    await openExtensionPage('options.html');
    await driver.wait(until.elementLocated({ css: 'form' }), 10_000);
    let stats: TabStats | null = null;
    await driver.wait(async () => {
      stats = await driver.executeAsyncScript<TabStats | null>((done: (v: unknown) => void) => {
        void (async () => {
          const all = await browser.storage.session.get(null);
          const entry = Object.entries(all).find(([key]) => key.startsWith('tab:'));
          if (!entry) return done(null);
          const total = { flagged: 0, clean: 0, pending: 0, failed: 0 };
          for (const frame of Object.values(entry[1] as Record<string, typeof total>)) {
            for (const key of Object.keys(total) as (keyof typeof total)[]) total[key] += frame[key];
          }
          const badgeText = await browser.action.getBadgeText({ tabId: Number(entry[0].slice(4)) });
          done({ ...total, badgeText });
        })();
      });
      return stats !== null && stats.pending === 0 && stats.flagged + stats.clean === 5;
    }, 20_000).catch(() => {});
    expect(stats).toMatchObject({ pending: 0, failed: 0 });
    expect(stats!.flagged).toBeGreaterThanOrEqual(4);
    expect(stats!.flagged + stats!.clean).toBe(5);
    expect(stats!.badgeText).toBe(String(stats!.flagged));
    await driver.close();
    await driver.switchTo().window(pageHandle);
  }, 60_000);

  it('scores an image without metadata with the bundled model', async () => {
    await openExtensionPage('options.html');
    await driver.wait(until.elementLocated({ css: 'form' }), 10_000);
    const verdict = await driver.executeAsyncScript<Record<string, unknown>>(
      (url: string, done: (v: unknown) => void) => {
        browser.runtime
          .sendMessage({ type: 'classify', url })
          .then(done, (e: Error) => done({ thrown: String(e) }));
      },
      `${origin}/plain.jpg`,
    );
    expect(verdict).toMatchObject({ ok: true, source: 'model' });
    expect(verdict.score).toBeGreaterThanOrEqual(0);
    expect(verdict.score).toBeLessThanOrEqual(1);
  }, 120_000);
});
