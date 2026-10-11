import { AutoModelForImageClassification, Tensor, env, type PreTrainedModel } from '@huggingface/transformers';
import model from '../../model.json';
import { toModelInput, type InputSpec, type PixelImage } from './preprocess';
import { aiLogit, calibratedProbability, type OutputSpec } from './score';

let loading: Promise<PreTrainedModel> | null = null;

function configureRuntime(): void {
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = browser.runtime.getURL('models/');
  env.useBrowserCache = false;
  env.useWasmCache = false;
  const wasm = env.backends.onnx.wasm!;
  wasm.wasmPaths = {
    mjs: browser.runtime.getURL('ort/ort-wasm-simd-threaded.asyncify.mjs'),
    wasm: browser.runtime.getURL('ort/ort-wasm-simd-threaded.asyncify.wasm'),
  };
  wasm.numThreads = 1;
}

function load(): Promise<PreTrainedModel> {
  loading ??= (async () => {
    configureRuntime();
    return AutoModelForImageClassification.from_pretrained(model.name, {
      dtype: model.dtype as 'q8',
      device: 'wasm',
    });
  })().catch((error) => {
    loading = null;
    throw error;
  });
  return loading;
}

export async function decodeImage(blob: Blob): Promise<PixelImage> {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d', { willReadFrequently: true })!;
    context.drawImage(bitmap, 0, 0);
    const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
    return { width: bitmap.width, height: bitmap.height, channels: 4, data };
  } finally {
    bitmap.close();
  }
}

export async function scoreImage(image: PixelImage): Promise<number> {
  const classifier = await load();
  const spec = model.input as InputSpec;
  const pixels = toModelInput(image, spec);
  const { logits } = await classifier({ pixel_values: new Tensor('float32', pixels, [1, 3, spec.crop, spec.crop]) });
  const id2label = (classifier.config as { id2label?: Record<string, string> }).id2label;
  return calibratedProbability(aiLogit(logits.data as Float32Array, model.output as OutputSpec, id2label), model.calibration);
}
