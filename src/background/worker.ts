import synthIdConfig from '../../synthid.json';
import type { Verdict } from '../shared/messages';
import { decodeImage, scoreImage } from './classifier';
import { detectAiMetadata } from './metadata';
import { isScorable, type PixelImage } from './preprocess';
import { SerialQueue } from './queue';
import { synthIdScore, type SynthIdTemplate } from './synthid';

export interface Checks {
  checkMetadata: boolean;
  checkWatermark: boolean;
}

export interface WorkerRequest extends Checks {
  id: number;
  bytes: ArrayBuffer;
  type: string;
}

export interface WorkerResponse {
  id: number;
  verdict: Verdict;
}

const scope = self as unknown as {
  postMessage(message: WorkerResponse): void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
};
const queue = new SerialQueue();
let template: Promise<SynthIdTemplate> | null = null;

function loadTemplate(): Promise<SynthIdTemplate> {
  template ??= fetch(new URL('synthid-template.bin', self.location.href))
    .then(async (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const bytes = new Uint8Array(await res.arrayBuffer());
      const count = bytes.length / 2;
      return {
        size: synthIdConfig.size,
        weightScale: synthIdConfig.weightScale,
        weights: bytes.subarray(0, count),
        phases: bytes.subarray(count),
      };
    })
    .catch((error) => {
      template = null;
      throw error;
    });
  return template;
}

async function hasSynthId(image: PixelImage): Promise<boolean> {
  try {
    return synthIdScore(image, await loadTemplate()) >= synthIdConfig.threshold;
  } catch (error) {
    console.error('[ai-watermark] SynthID check failed', error);
    return false;
  }
}

async function score(bytes: Uint8Array, type: string, checkWatermark: boolean): Promise<Verdict> {
  let image;
  try {
    image = await decodeImage(new Blob([bytes as BlobPart], { type }));
  } catch {
    return { ok: false, error: 'decode' };
  }
  if (checkWatermark && (await hasSynthId(image))) return { ok: true, score: 1, source: 'synthid' };
  if (!isScorable(image)) return { ok: false, error: 'unsupported' };
  try {
    return { ok: true, score: await scoreImage(image), source: 'model' };
  } catch (error) {
    console.error('[ai-watermark] inference failed', error);
    return { ok: false, error: 'model' };
  }
}

scope.onmessage = ({ data }) => {
  const bytes = new Uint8Array(data.bytes);
  const reason = data.checkMetadata ? detectAiMetadata(bytes) : null;
  const verdict: Promise<Verdict> = reason
    ? Promise.resolve({ ok: true, score: 1, source: 'metadata', reason })
    : queue.run(() => score(bytes, data.type, data.checkWatermark));
  void verdict.then((v) => scope.postMessage({ id: data.id, verdict: v }));
};
