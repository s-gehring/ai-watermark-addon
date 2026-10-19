import * as ort from 'onnxruntime-web/wasm';
import config from '../../model.json';
import { extraViews, inputShape, toModelInput, type InputSpec, type PixelImage } from './preprocess';
import { aiLogit, calibratedProbability, modelProbabilities, type OutputSpec } from './score';

type ModelEntry = (typeof config.models)[number];

interface Classifier {
  session: ort.InferenceSession;
  id2label?: Record<string, string>;
}

let loading: Promise<Classifier[]> | null = null;

const assetUrl = (path: string) => new URL(path, self.location.href).href;

const onnxFile = (model: ModelEntry) => Object.keys(model.files).find((file) => file.endsWith('.onnx'))!;

async function loadClassifier(model: ModelEntry): Promise<Classifier> {
  const session = await ort.InferenceSession.create(assetUrl(`models/${model.name}/${onnxFile(model)}`), {
    executionProviders: ['wasm'],
  });
  if (!('config.json' in model.files)) return { session };
  const res = await fetch(assetUrl(`models/${model.name}/config.json`));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return { session, id2label: (await res.json()).id2label };
}

function load(): Promise<Classifier[]> {
  loading ??= (async () => {
    ort.env.wasm.wasmPaths = { wasm: assetUrl('ort/ort-wasm-simd-threaded.wasm') };
    ort.env.wasm.numThreads = 1;
    return Promise.all(config.models.map(loadClassifier));
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

async function modelLogit(classifier: Classifier, model: ModelEntry, input: Float32Array): Promise<number> {
  const [width, height] = inputShape(model.input as InputSpec);
  const { session, id2label } = classifier;
  const output = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, height, width]) });
  return aiLogit(output[session.outputNames[0]].data as Float32Array, model.output as OutputSpec, id2label);
}

export async function scoreImage(image: PixelImage): Promise<{ score: number; models: number[] }> {
  const classifiers = await load();
  const logits: number[] = [];
  for (const [i, model] of config.models.entries()) {
    logits.push(await modelLogit(classifiers[i], model, toModelInput(image, model.input as InputSpec)));
  }
  const { cascade } = config;
  const score = calibratedProbability(logits, config.calibration);
  if (config.models.length === 1 && score >= cascade.band[0] && score < cascade.band[1]) {
    const [model] = config.models;
    for (const input of extraViews(image, model.input as InputSpec, cascade.shortestEdges)) {
      logits[0] = Math.max(logits[0], await modelLogit(classifiers[0], model, input));
    }
  }
  return { score: calibratedProbability(logits, config.calibration), models: modelProbabilities(logits, config.calibration) };
}
