import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PixelImage } from '../../src/background/preprocess';
import type { WorkerRequest, WorkerResponse } from '../../src/background/worker';

const classifier = vi.hoisted(() => ({ activeBackend: () => 'wasm', decodeImage: vi.fn(), reencodeAsJpeg: vi.fn(), scoreImage: vi.fn() }));
const synthid = vi.hoisted(() => ({ detectSynthId: vi.fn(), searchCroppedSynthId: vi.fn() }));
const noMark = vi.hoisted(() => () => null);
const invisible = vi.hoisted(() => ({ detectInvisibleWatermark: vi.fn() }));
const metadata = vi.hoisted(() => ({ detectAiMetadata: vi.fn() }));

vi.mock('../../src/background/classifier', () => classifier);
vi.mock('../../src/background/synthid', () => synthid);
vi.mock('../../src/background/invisible-watermark', () => invisible);
vi.mock('../../src/background/metadata', () => metadata);
vi.mock('../../src/background/bytedance-watermark', () => ({ detectBytedanceWatermark: noMark }));
vi.mock('../../src/background/visible-watermark', () => ({ detectVisibleWatermark: noMark }));

const REENCODED = { width: 1, height: 1, channels: 4, data: [0, 0, 0, 255] } satisfies PixelImage;
const scope = { postMessage: vi.fn(), onmessage: null as ((event: { data: WorkerRequest }) => void) | null, location: { href: 'https://worker.test/worker.js' } };
let nextId = 0;

const encoded = (width: number, height: number) => ({ bytes: new Uint16Array([width, height]).buffer as ArrayBuffer, type: 'image/png' });

function run(displayed: [number, number], original: [number, number] | null, checkWatermark = false, checkMetadata = false): Promise<WorkerResponse> {
  const id = nextId++;
  return new Promise((resolve) => {
    scope.postMessage.mockImplementation((response: WorkerResponse) => response.id === id && resolve(response));
    scope.onmessage!({
      data: { id, ...encoded(...displayed), original: original && encoded(...original), checkMetadata, checkWatermark },
    });
  });
}

const sizeOf = ({ width, height }: PixelImage) => [width, height];

beforeAll(async () => {
  vi.stubGlobal('self', scope);
  vi.stubGlobal('fetch', async () => new Response(new Uint8Array(2)));
  await import('../../src/background/worker');
});

beforeEach(() => {
  vi.clearAllMocks();
  classifier.decodeImage.mockImplementation(async (blob: Blob): Promise<PixelImage> => {
    const [width, height] = new Uint16Array(await blob.arrayBuffer());
    return { width, height, channels: 4, data: new Uint8ClampedArray(width * height * 4) };
  });
  classifier.reencodeAsJpeg.mockResolvedValue(REENCODED);
  classifier.scoreImage.mockResolvedValue({ score: 0.1, models: [0.1] });
  synthid.detectSynthId.mockReturnValue(null);
  synthid.searchCroppedSynthId.mockReturnValue(null);
  invisible.detectInvisibleWatermark.mockReturnValue(null);
  metadata.detectAiMetadata.mockReturnValue(null);
});

describe('worker', () => {
  it('scores the larger variant of a thumbnail, reduced to 800 px and re-encoded as JPEG 80', async () => {
    const { verdict } = await run([400, 300], [1600, 1200]);
    expect(verdict).toMatchObject({ ok: true, source: 'model', score: 0.1 });
    const [reduced, quality] = classifier.reencodeAsJpeg.mock.calls[0];
    expect(sizeOf(reduced)).toEqual([800, 600]);
    expect(quality).toBe(0.8);
    expect(classifier.scoreImage).toHaveBeenCalledExactlyOnceWith(REENCODED);
  });

  it('keeps the unmodified variant for the watermark checks', async () => {
    await run([400, 300], [1600, 1200], true);
    expect(sizeOf(synthid.detectSynthId.mock.calls[0][0])).toEqual([1600, 1200]);
    expect(classifier.scoreImage).toHaveBeenCalledExactlyOnceWith(REENCODED);
  });

  it.each([
    ['is not a thumbnail', [800, 600], [1600, 1200]],
    ['has no larger variant', [400, 300], null],
    ['has a variant of another aspect ratio', [400, 300], [1600, 900]],
    ['has a variant of the same size', [400, 300], [400, 300]],
  ] as const)('scores the displayed image when it %s', async (_, displayed, original) => {
    await run([...displayed], original && [...original]);
    expect(classifier.reencodeAsJpeg).not.toHaveBeenCalled();
    expect(sizeOf(classifier.scoreImage.mock.calls[0][0])).toEqual([...displayed]);
  });

  it('answers with a model error when a check throws, and keeps answering afterwards', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    invisible.detectInvisibleWatermark.mockImplementationOnce(() => {
      throw new Error('broken check');
    });
    expect((await run([400, 300], null, true)).verdict).toEqual({ ok: false, error: 'model' });
    metadata.detectAiMetadata.mockImplementationOnce(() => {
      throw new Error('broken parser');
    });
    expect((await run([400, 300], null, false, true)).verdict).toEqual({ ok: false, error: 'model' });
    expect(logged).toHaveBeenCalledTimes(2);
    logged.mockRestore();
    expect((await run([400, 300], null, true)).verdict).toMatchObject({ ok: true, source: 'model' });
  });
});
