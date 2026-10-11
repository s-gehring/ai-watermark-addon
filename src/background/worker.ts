import synthIdConfig from '../../synthid.json';
import type { SynthIdProvider, Verdict } from '../shared/messages';
import { decodeImage, scoreImage } from './classifier';
import { detectAiMetadata } from './metadata';
import { isScorable, type PixelImage } from './preprocess';
import { SerialQueue } from './queue';
import { detectSynthId, type SynthIdDetector } from './synthid';

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
let detectors: Promise<SynthIdDetector[]> | null = null;

async function loadDetector(entry: (typeof synthIdConfig.templates)[number]): Promise<SynthIdDetector> {
  const res = await fetch(new URL(`synthid-templates/${entry.file}`, self.location.href));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  const count = bytes.length / 2;
  return {
    provider: entry.provider as SynthIdProvider,
    threshold: entry.threshold,
    aspect: entry.outputSize && entry.outputSize[0] / entry.outputSize[1],
    size: entry.size,
    weightScale: entry.weightScale,
    weights: bytes.subarray(0, count),
    phases: bytes.subarray(count),
  };
}

function loadDetectors(): Promise<SynthIdDetector[]> {
  detectors ??= Promise.all(synthIdConfig.templates.map(loadDetector)).catch((error) => {
    detectors = null;
    throw error;
  });
  return detectors;
}

async function findSynthId(image: PixelImage): Promise<SynthIdProvider | null> {
  try {
    return detectSynthId(image, await loadDetectors());
  } catch (error) {
    console.error('[ai-watermark] SynthID check failed', error);
    return null;
  }
}

async function score(bytes: Uint8Array, type: string, checkWatermark: boolean): Promise<Verdict> {
  let image;
  try {
    image = await decodeImage(new Blob([bytes as BlobPart], { type }));
  } catch {
    return { ok: false, error: 'decode' };
  }
  const provider = checkWatermark ? await findSynthId(image) : null;
  if (provider) return { ok: true, score: 1, source: 'synthid', provider };
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
