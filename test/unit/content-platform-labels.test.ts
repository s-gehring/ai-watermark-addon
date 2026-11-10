import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PlatformLabels } from '../../src/content/platform-labels';

const fixture = (name: string) => readFileSync(new URL(`./fixtures/platform-labels/${name}`, import.meta.url), 'utf8');
const PIXIV_AI = 'https://i.pximg.net/c/360x360_70/img-master/img/2025/10/20/17/13/22/136500000_p0_square1200.jpg';
const PIXIV_OTHER = 'https://i.pximg.net/c/360x360_70/custom-thumb/img/2024/12/08/11/40/51/125000000_p0_custom1200.jpg';
const PIXIV_AVATAR = 'https://i.pximg.net/user-profile/img/2024/03/01/10/20/30/25600000_0123456789abcdef_170.jpg';

interface Page {
  url: string;
  scripts?: string[];
  categories?: string[];
}

const sendMessage = vi.fn();

function open({ url, scripts = [], categories = [] }: Page): void {
  vi.stubGlobal('location', new URL(url));
  vi.stubGlobal('document', {
    scripts: scripts.map((textContent) => ({ src: '', textContent })),
    querySelectorAll: () => categories.map((textContent) => ({ textContent })),
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  sendMessage.mockReset();
  vi.stubGlobal('window', { setTimeout });
  vi.stubGlobal('browser', { runtime: { sendMessage } });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('PlatformLabels', () => {
  it('does nothing on other sites', () => {
    open({ url: 'https://example.com/gallery' });
    expect(PlatformLabels.forPage(() => [])).toBeNull();
  });

  it('asks the background once for all pixiv works found together and labels only the AI works', async () => {
    open({ url: 'https://www.pixiv.net/artworks/136500000' });
    sendMessage.mockResolvedValue(['136500000']);
    const labels = PlatformLabels.forPage(() => [])!;
    const verdicts = [labels.verdictFor(PIXIV_AI), labels.verdictFor(PIXIV_OTHER), labels.verdictFor(PIXIV_AI)];
    expect(labels.verdictFor(PIXIV_AVATAR)).toBeNull();
    await vi.runAllTimersAsync();
    expect(await Promise.all(verdicts)).toEqual([
      { ok: true, score: 1, source: 'platform', platform: 'pixiv' },
      null,
      { ok: true, score: 1, source: 'platform', platform: 'pixiv' },
    ]);
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
    open({ url: 'https://www.pixiv.net/ranking.php?mode=daily' });
    sendMessage.mockResolvedValue(['125000000']);
    const labels = PlatformLabels.forPage(() => [PIXIV_AI, PIXIV_AVATAR, PIXIV_OTHER])!;
    const verdict = labels.verdictFor(PIXIV_AI);
    await vi.runAllTimersAsync();
    expect(await verdict).toBeNull();
    expect(sendMessage.mock.calls[0][0]).toMatchObject({ keys: ['136500000', '125000000'] });
    expect(await labels.verdictFor(PIXIV_OTHER)).toMatchObject({ source: 'platform', platform: 'pixiv' });
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it('treats a failed lookup as no label', async () => {
    open({ url: 'https://www.pixiv.net/' });
    sendMessage.mockRejectedValue(new Error('no background'));
    const verdict = PlatformLabels.forPage(() => [])!.verdictFor(PIXIV_AI);
    await vi.runAllTimersAsync();
    expect(await verdict).toBeNull();
  });

  it('reads DeviantArt labels from the page without any request', async () => {
    const html = fixture('deviantart-tag-dreamup.html');
    open({ url: 'https://www.deviantart.com/tag/dreamup', scripts: ['var other = 1;', /<script>([\s\S]*)<\/script>/.exec(html)![1]] });
    const labels = PlatformLabels.forPage(() => [])!;
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
    sendMessage.mockResolvedValue([]);
    const labels = PlatformLabels.forPage(() => [])!;
    const own = labels.verdictFor('https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Vaporwave_sunset.jpg/960px-Vaporwave_sunset.jpg');
    const other = labels.verdictFor('https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Other_file.jpg/120px-Other_file.jpg');
    expect(await own).toMatchObject({ source: 'platform', platform: 'commons' });
    await vi.runAllTimersAsync();
    expect(await other).toBeNull();
    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage.mock.calls[0][0]).toMatchObject({ platform: 'commons', keys: ['File:Other file.jpg'] });
  });
});
