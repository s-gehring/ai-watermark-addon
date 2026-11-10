import { parentPort, workerData } from 'node:worker_threads';
import { detectBytedanceWatermark } from '../../src/background/bytedance-watermark';
import { detectInvisibleWatermark } from '../../src/background/invisible-watermark';
import { detectAiMetadata } from '../../src/background/metadata';
import type { PixelImage } from '../../src/background/preprocess';
import {
  detectSynthId,
  searchCroppedSynthId,
  type CropSearchConfig,
  type SpectrumCache,
  type SynthIdDetector,
} from '../../src/background/synthid';
import { detectVisibleWatermark } from '../../src/background/visible-watermark';

type Hit = ['metadata' | 'synthid' | 'watermark', unknown] | null;

interface Result {
  hit: Hit;
  crop?: string | null;
}

interface Task {
  bytes: Uint8Array;
  image: PixelImage;
  detectorSet: number;
  searchCrop: boolean;
  known?: Result;
}

const { detectorSets, cropSearch } = workerData as { detectorSets: SynthIdDetector[][]; cropSearch: CropSearchConfig };

function firstHit(bytes: Uint8Array, image: PixelImage, detectors: SynthIdDetector[], spectra: SpectrumCache): Hit {
  const reason = detectAiMetadata(bytes);
  if (reason) return ['metadata', reason];
  const provider = detectSynthId(image, detectors, spectra);
  if (provider) return [provider === 'amazon' ? 'watermark' : 'synthid', provider];
  const scheme = detectInvisibleWatermark(image) ?? detectBytedanceWatermark(image) ?? detectVisibleWatermark(image);
  return scheme ? ['watermark', scheme] : null;
}

parentPort!.on('message', ({ bytes, image, detectorSet, searchCrop, known }: Task) => {
  try {
    const detectors = detectorSets[detectorSet];
    const spectra: SpectrumCache = new Map();
    const result: Result = known ?? { hit: firstHit(bytes, image, detectors, spectra) };
    if (searchCrop && result.hit === null && result.crop === undefined) {
      result.crop = searchCroppedSynthId(image, detectors, cropSearch, spectra);
    }
    parentPort!.postMessage({ result });
  } catch (error) {
    parentPort!.postMessage({ error: (error as Error).message });
  }
});
