import { AutoModelForImageClassification, Tensor, env, type PreTrainedModel } from '@huggingface/transformers';
import config from '../../model.json';
import { inputShape, toModelInput, type InputSpec, type PixelImage } from './preprocess';
import { aiLogit, calibratedProbability, modelProbabilities, type OutputSpec } from './score';

let loading: Promise<PreTrainedModel[]> | null = null;

const assetUrl = (path: string) => new URL(path, self.location.href).href;

function configureRuntime(): void {
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = assetUrl('models/');
  env.useBrowserCache = false;
  env.useWasmCache = false;
  const wasm = env.backends.onnx.wasm!;
  wasm.wasmPaths = {
    mjs: assetUrl('ort/ort-wasm-simd-threaded.asyncify.mjs'),
    wasm: assetUrl('ort/ort-wasm-simd-threaded.asyncify.wasm'),
  };
  wasm.numThreads = 1;
}

function load(): Promise<PreTrainedModel[]> {
  loading ??= (async () => {
    configureRuntime();
    return Promise.all(
      config.models.map((model) =>
        AutoModelForImageClassification.from_pretrained(model.name, { dtype: model.dtype as 'q8', device: 'wasm' }),
      ),
    );
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

export async function scoreImage(image: PixelImage): Promise<{ score: number; models: number[] }> {
  const classifiers = await load();
  const logits: number[] = [];
  for (const [i, model] of config.models.entries()) {
    const spec = model.input as InputSpec;
    const [width, height] = inputShape(spec);
    const pixels = toModelInput(image, spec);
    const output = await classifiers[i]({ pixel_values: new Tensor('float32', pixels, [1, 3, height, width]) });
    const id2label = (classifiers[i].config as { id2label?: Record<string, string> }).id2label;
    logits.push(aiLogit(output.logits.data as Float32Array, model.output as OutputSpec, id2label));
  }
  return { score: calibratedProbability(logits, config.calibration), models: modelProbabilities(logits, config.calibration) };
}
