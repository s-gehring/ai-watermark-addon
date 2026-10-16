import synthIdConfig from '../../synthid.json';
import type { Verdict } from '../shared/messages';
import { decodeImage, scoreImage } from './classifier';
import { detectInvisibleWatermark } from './invisible-watermark';
import { detectAiMetadata } from './metadata';
import { isScorable, type PixelImage } from './preprocess';
import { SerialQueue } from './queue';
import { detectSynthId, type SynthIdDetector, type TemplateProvider } from './synthid';
import { detectVisibleWatermark } from './visible-watermark';

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
    provider: entry.provider as TemplateProvider,
    threshold: entry.threshold,
    aspect: entry.outputSize && entry.outputSize[0] / entry.outputSize[1],
    size: entry.size,
    weightScale: entry.weightScale,
    minChannelStd: entry.minChannelStd,
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

async function findSynthId(image: PixelImage): Promise<TemplateProvider | null> {
  try {
    return detectSynthId(image, await loadDetectors());
  } catch (error) {
    console.error('[ai-watermark] SynthID check failed', error);
    return null;
  }
}

async function score(bytes: Uint8Array, type: string, checks: Checks): Promise<Verdict> {
  let image;
  try {
    image = await decodeImage(new Blob([bytes as BlobPart], { type }));
  } catch {
    return { ok: false, error: 'decode' };
  }
  const metadata = checks.checkMetadata ? 'clear' : 'off';
  const provider = checks.checkWatermark ? await findSynthId(image) : null;
  if (provider === 'amazon') return { ok: true, score: 1, source: 'watermark', scheme: 'nova-canvas', metadata };
  if (provider) return { ok: true, score: 1, source: 'synthid', provider, metadata };
  const scheme = checks.checkWatermark ? (detectInvisibleWatermark(image) ?? detectVisibleWatermark(image)) : null;
  if (scheme) return { ok: true, score: 1, source: 'watermark', scheme, metadata };
  if (!isScorable(image)) return { ok: false, error: 'unsupported' };
  try {
    const watermark = checks.checkWatermark ? 'clear' : 'off';
    return { ok: true, ...(await scoreImage(image)), source: 'model', metadata, watermark };
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
    : queue.run(() => score(bytes, data.type, data));
  void verdict.then((v) => scope.postMessage({ id: data.id, verdict: v }));
};
