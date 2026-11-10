import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Verdict } from '../../src/shared/messages';
import { labelFirst, PlatformLabels } from '../../src/content/platform-labels';

const fixture = (name: string) => readFileSync(new URL(`./fixtures/platform-labels/${name}`, import.meta.url), 'utf8');
const PIXIV_AI = 'https://i.pximg.net/c/360x360_70/img-master/img/2025/10/20/17/13/22/136500000_p0_square1200.jpg';
const PIXIV_AI_LARGE = 'https://i.pximg.net/img-master/img/2025/10/20/17/13/22/136500000_p0_master1200.jpg';
const PIXIV_OTHER = 'https://i.pximg.net/c/360x360_70/custom-thumb/img/2024/12/08/11/40/51/125000000_p0_custom1200.jpg';
const PIXIV_AVATAR = 'https://i.pximg.net/user-profile/img/2024/03/01/10/20/30/25600000_0123456789abcdef_170.jpg';
const LABELED: Verdict = { ok: true, score: 1, source: 'platform', platform: 'pixiv' };
const CLEAN: Verdict = { ok: true, score: 0.02, source: 'model', models: [0.02], metadata: 'clear', watermark: 'clear' };

interface Page {
  url: string;
  scripts?: string[];
  categories?: string[];
  images?: { src: string; large: boolean }[];
}

const sendMessage = vi.fn();
const isLarge = (image: HTMLImageElement) => (image as unknown as { large: boolean }).large;

function open({ url, scripts = [], categories = [], images = [] }: Page): void {
  vi.stubGlobal('location', new URL(url));
  vi.stubGlobal('document', {
    scripts: scripts.map((textContent) => ({ src: '', textContent })),
    images: images.map((image) => ({ ...image, currentSrc: '' })),
    querySelectorAll: () => categories.map((textContent) => ({ textContent })),
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  sendMessage.mockReset();
  vi.stubGlobal('window', { setTimeout, clearTimeout });
  vi.stubGlobal('browser', { runtime: { sendMessage } });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('PlatformLabels', () => {
  it('does nothing on other sites', () => {
    open({ url: 'https://example.com/gallery' });
    expect(PlatformLabels.forPage(isLarge)).toBeNull();
  });

  it('asks the background once for all pixiv works found together and labels only the AI works', async () => {
    open({ url: 'https://www.pixiv.net/artworks/136500000' });
    sendMessage.mockResolvedValue({ 136500000: true, 125000000: false });
    const labels = PlatformLabels.forPage(isLarge)!;
    const verdicts = [labels.verdictFor(PIXIV_AI), labels.verdictFor(PIXIV_OTHER), labels.verdictFor(PIXIV_AI)];
    expect(labels.verdictFor(PIXIV_AVATAR)).toBeNull();
    await vi.runAllTimersAsync();
    expect(await Promise.all(verdicts)).toEqual([LABELED, null, LABELED]);
    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage).toHaveBeenCalledWith({
      type: 'platform-labels',
      platform: 'pixiv',
      keys: ['136500000', '125000000'],
      page: 'https://www.pixiv.net/artworks/136500000',
    });
    await labels.verdictFor(PIXIV_OTHER);
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it('asks in the same request for the other large works on the page', async () => {
    open({
      url: 'https://www.pixiv.net/ranking.php?mode=daily',
      images: [
        { src: PIXIV_AI, large: true },
        { src: PIXIV_AVATAR, large: true },
        { src: PIXIV_OTHER, large: true },
        { src: PIXIV_AI.replace('136500000', '111'), large: false },
      ],
    });
    sendMessage.mockResolvedValue({ 136500000: false, 125000000: true });
    const labels = PlatformLabels.forPage(isLarge)!;
    const verdict = labels.verdictFor(PIXIV_AI);
    await vi.runAllTimersAsync();
    expect(await verdict).toBeNull();
    expect(sendMessage.mock.calls[0][0]).toMatchObject({ keys: ['136500000', '125000000'] });
    expect(await labels.verdictFor(PIXIV_OTHER)).toEqual(LABELED);
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it('asks again on the next page view for a work whose lookup failed', async () => {
    open({ url: 'https://www.pixiv.net/ranking.php?mode=daily_ai' });
    sendMessage.mockRejectedValueOnce(new Error('no background'));
    const labels = PlatformLabels.forPage(isLarge)!;
    const failed = labels.verdictFor(PIXIV_AI);
    await vi.runAllTimersAsync();
    expect(await failed).toBeNull();
    expect(await labels.verdictFor(PIXIV_AI)).toBeNull();
    expect(sendMessage).toHaveBeenCalledTimes(1);

    open({ url: 'https://www.pixiv.net/artworks/136500000' });
    sendMessage.mockResolvedValue({ 136500000: true });
    const again = labels.verdictFor(PIXIV_AI_LARGE);
    await vi.runAllTimersAsync();
    expect(await again).toEqual(LABELED);
    expect(sendMessage).toHaveBeenCalledTimes(2);
    expect(sendMessage.mock.calls[1][0]).toMatchObject({ keys: ['136500000'], page: 'https://www.pixiv.net/artworks/136500000' });
  });

  it('asks again right away when the page changed while an unanswered lookup was running', async () => {
    open({ url: 'https://www.pixiv.net/ranking.php?mode=daily_ai' });
    let answer!: (answers: Record<string, boolean>) => void;
    sendMessage.mockReturnValueOnce(new Promise((resolve) => (answer = resolve)));
    sendMessage.mockResolvedValue({ 136500000: true });
    const labels = PlatformLabels.forPage(isLarge)!;
    const verdict = labels.verdictFor(PIXIV_AI);
    await vi.advanceTimersByTimeAsync(100);
    open({ url: 'https://www.pixiv.net/artworks/136500000' });
    answer({});
    await vi.runAllTimersAsync();
    expect(await verdict).toEqual(LABELED);
    expect(sendMessage).toHaveBeenCalledTimes(2);
  });

  it('reads DeviantArt labels from the page without any request', async () => {
    const html = fixture('deviantart-tag-dreamup.html');
    open({ url: 'https://www.deviantart.com/tag/dreamup', scripts: ['var other = 1;', /<script>([\s\S]*)<\/script>/.exec(html)![1]] });
    const labels = PlatformLabels.forPage(isLarge)!;
    const file = 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/2da1c90a-37f6-48a9-b3f6-4e05f9a8273e/dfi325b-359f7dc4-83b4-4e4b-b642-c00659e1fc14.jpg';
    expect(await labels.verdictFor(`${file}/v1/fill/w_300,h_200,q_70,strp/x.jpg?token=t`)).toMatchObject({ source: 'platform', platform: 'deviantart' });
    expect(await labels.verdictFor(file.replace('2da1c90a', '00000000'))).toBeNull();
    expect(labels.verdictFor('https://a.deviantart.net/avatars-big/s/o/someone.jpg?8')).toBeNull();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('takes the label of a Commons file page from its category links', async () => {
    open({
      url: 'https://commons.wikimedia.org/wiki/File:Vaporwave_sunset.jpg',
      categories: ['AI-generated images', 'Vaporwave', 'CC-BY-3.0'],
    });
    sendMessage.mockResolvedValue({ 'File:Other file.jpg': false });
    const labels = PlatformLabels.forPage(isLarge)!;
    const own = labels.verdictFor('https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Vaporwave_sunset.jpg/960px-Vaporwave_sunset.jpg');
    const other = labels.verdictFor('https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Other_file.jpg/120px-Other_file.jpg');
    expect(await own).toMatchObject({ source: 'platform', platform: 'commons' });
    await vi.runAllTimersAsync();
    expect(await other).toBeNull();
    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage.mock.calls[0][0]).toMatchObject({ platform: 'commons', keys: ['File:Other file.jpg'] });
  });
});

describe('labelFirst', () => {
  const GRACE = 400;
  const later = <T>(value: T, ms: number) => new Promise<T>((resolve) => setTimeout(() => resolve(value), ms));

  it('runs the normal check at once where there is no platform label', async () => {
    const show = vi.fn();
    const check = vi.fn(async () => CLEAN);
    labelFirst(null, check, GRACE, show);
    expect(check).toHaveBeenCalledTimes(1);
    await vi.runAllTimersAsync();
    expect(show.mock.calls).toEqual([[CLEAN]]);
  });

  it('shows a label that arrives in time and skips the normal check', async () => {
    const show = vi.fn();
    const check = vi.fn(async () => CLEAN);
    labelFirst(later(LABELED, 200), check, GRACE, show);
    await vi.runAllTimersAsync();
    expect(show.mock.calls).toEqual([[LABELED]]);
    expect(check).not.toHaveBeenCalled();
  });

  it('runs the normal check as soon as the work turns out to be unlabeled', async () => {
    const show = vi.fn();
    const check = vi.fn(async () => CLEAN);
    labelFirst(later(null, 200), check, GRACE, show);
    await vi.advanceTimersByTimeAsync(199);
    expect(check).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(check).toHaveBeenCalledTimes(1);
    await vi.runAllTimersAsync();
    expect(show.mock.calls).toEqual([[CLEAN]]);
  });

  it('does not wait longer than the grace period and lets a late label replace the result', async () => {
    const show = vi.fn();
    const check = vi.fn(() => later(CLEAN, 300));
    labelFirst(later(LABELED, 2000), check, GRACE, show);
    await vi.advanceTimersByTimeAsync(GRACE);
    expect(check).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(300);
    expect(show.mock.calls).toEqual([[CLEAN]]);
    await vi.runAllTimersAsync();
    expect(show.mock.calls).toEqual([[CLEAN], [LABELED]]);
    expect(check).toHaveBeenCalledTimes(1);
  });

  it('keeps a late label when the normal check finishes after it', async () => {
    const show = vi.fn();
    labelFirst(later(LABELED, 600), () => later(CLEAN, 5000), GRACE, show);
    await vi.runAllTimersAsync();
    expect(show.mock.calls).toEqual([[LABELED]]);
  });
});
