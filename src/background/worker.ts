import type { Verdict } from '../shared/messages';
import { decodeImage, scoreImage } from './classifier';
import { detectAiMetadata } from './metadata';
import { isScorable } from './preprocess';
import { SerialQueue } from './queue';

export interface WorkerRequest {
  id: number;
  bytes: ArrayBuffer;
  type: string;
  checkMetadata: boolean;
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

async function score(bytes: Uint8Array, type: string): Promise<Verdict> {
  let image;
  try {
    image = await decodeImage(new Blob([bytes as BlobPart], { type }));
  } catch {
    return { ok: false, error: 'decode' };
  }
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
    : queue.run(() => score(bytes, data.type));
  void verdict.then((v) => scope.postMessage({ id: data.id, verdict: v }));
};
