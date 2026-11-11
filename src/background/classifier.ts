import * as wasmOrt from 'onnxruntime-web/wasm';
import * as gpuOrt from 'onnxruntime-web/jspi';
import config from '../../model.json';
import { extraViews, inputShape, selfTestInput, toModelInput, type InputSpec, type PixelImage } from './preprocess';
import { aiLogit, calibratedProbability, modelProbabilities, type OutputSpec } from './score';

type ModelEntry = (typeof config.models)[number];
type Ort = typeof wasmOrt;
export type Backend = 'webgpu' | 'wasm';

interface Classifier {
  session: wasmOrt.InferenceSession;
  id2label?: Record<string, string>;
}

interface Runtime {
  backend: Backend;
  ort: Ort;
  classifiers: Classifier[];
}

export const GPU_PLATFORMS = ['Mac'];
export const GPU_INIT_TIMEOUT_MS = 15_000;
export const SELF_TEST_TOLERANCE = 0.01;

let loading: Promise<Runtime> | null = null;
let active: Runtime | null = null;
let gpuFailed = false;

const assetUrl = (path: string) => new URL(path, self.location.href).href;

const onnxFile = (model: ModelEntry) => Object.keys(model.files).find((file) => file.endsWith('.onnx'))!;

async function loadClassifier(ort: Ort, backend: Backend, model: ModelEntry): Promise<Classifier> {
  const session = await ort.InferenceSession.create(assetUrl(`models/${model.name}/${onnxFile(model)}`), {
    executionProviders: [backend],
  });
  if (!('config.json' in model.files)) return { session };
  const res = await fetch(assetUrl(`models/${model.name}/config.json`));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return { session, id2label: (await res.json()).id2label };
}

async function createRuntime(ort: Ort, backend: Backend, wasmFile: string): Promise<Runtime> {
  ort.env.wasm.wasmPaths = { wasm: assetUrl(`ort/${wasmFile}`) };
  ort.env.wasm.numThreads = 1;
  return { backend, ort, classifiers: await Promise.all(config.models.map((model) => loadClassifier(ort, backend, model))) };
}

function release(runtime: Runtime): void {
  for (const { session } of runtime.classifiers) void session.release().catch(() => {});
}

function wantsGpu(): boolean {
  return (
    !gpuFailed &&
    GPU_PLATFORMS.some((platform) => navigator.platform.startsWith(platform)) &&
    'gpu' in navigator &&
    'Suspending' in WebAssembly &&
    new URL(self.location.href).searchParams.get('backend') !== 'wasm'
  );
}

async function loadGpu(): Promise<Runtime> {
  const runtime = await createRuntime(gpuOrt as unknown as Ort, 'webgpu', 'ort-wasm-simd-threaded.jspi.wasm');
  try {
    for (const [i, model] of config.models.entries()) {
      const [width, height] = inputShape(model.input as InputSpec);
      const logit = await modelLogit(runtime, i, selfTestInput(width, height));
      if (!(Math.abs(logit - model.selfTestLogit) <= SELF_TEST_TOLERANCE)) {
        throw new Error(`self-test logit ${logit}, expected ${model.selfTestLogit}`);
      }
    }
  } catch (error) {
    release(runtime);
    throw error;
  }
  void Promise.resolve(gpuOrt.env.webgpu.device as Promise<GPUDevice> | GPUDevice | undefined)
    .then((device) => device?.lost)
    .then((info) => info && abandonGpu(runtime, `device lost: ${info.message}`));
  return runtime;
}

function withTimeout<T>(promise: Promise<T>, ms: number, onLate: (value: T) => void): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      void promise.then(onLate, () => {});
      reject(new Error(`no answer within ${ms} ms`));
    }, ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function abandonGpu(runtime: Runtime, reason: unknown): void {
  if (active === runtime) {
    active = null;
    loading = null;
  }
  if (gpuFailed) return;
  gpuFailed = true;
  console.warn('[ai-watermark] WebGPU disabled, using WASM:', reason);
  release(runtime);
}

async function startRuntime(): Promise<Runtime> {
  if (wantsGpu()) {
    try {
      return await withTimeout(loadGpu(), GPU_INIT_TIMEOUT_MS, release);
    } catch (error) {
      gpuFailed = true;
      console.warn('[ai-watermark] WebGPU unavailable, using WASM:', error);
    }
  }
  return createRuntime(wasmOrt, 'wasm', 'ort-wasm-simd-threaded.wasm');
}

function load(): Promise<Runtime> {
  loading ??= startRuntime().then(
    (runtime) => {
      if (runtime.backend !== 'webgpu' || !gpuFailed) return (active = runtime);
      loading = null;
      return load();
    },
    (error) => {
      loading = null;
      throw error;
    },
  );
  return loading;
}

export function activeBackend(): Backend | null {
  return active?.backend ?? null;
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

export async function reencodeAsJpeg(image: PixelImage, quality: number): Promise<PixelImage> {
  const canvas = new OffscreenCanvas(image.width, image.height);
  canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(image.data), image.width, image.height), 0, 0);
  return decodeImage(await canvas.convertToBlob({ type: 'image/jpeg', quality }));
}

async function modelLogit(runtime: Runtime, index: number, input: Float32Array): Promise<number> {
  const model = config.models[index];
  const [width, height] = inputShape(model.input as InputSpec);
  const { session, id2label } = runtime.classifiers[index];
  const tensor = new runtime.ort.Tensor('float32', input, [1, 3, height, width]);
  const output = await session.run({ [session.inputNames[0]]: tensor });
  return aiLogit(output[session.outputNames[0]].data as Float32Array, model.output as OutputSpec, id2label);
}

async function scoreWith(runtime: Runtime, image: PixelImage): Promise<{ score: number; models: number[] }> {
  const logits: number[] = [];
  for (const [i, model] of config.models.entries()) {
    logits.push(await modelLogit(runtime, i, toModelInput(image, model.input as InputSpec)));
  }
  const { cascade } = config;
  const score = calibratedProbability(logits, config.calibration);
  if (config.models.length === 1 && score >= cascade.band[0] && score < cascade.band[1]) {
    const [model] = config.models;
    for (const input of extraViews(image, model.input as InputSpec, cascade.shortestEdges)) {
      logits[0] = Math.max(logits[0], await modelLogit(runtime, 0, input));
    }
  }
  return { score: calibratedProbability(logits, config.calibration), models: modelProbabilities(logits, config.calibration) };
}

export async function scoreImage(image: PixelImage): Promise<{ score: number; models: number[] }> {
  const runtime = await load();
  try {
    return await scoreWith(runtime, image);
  } catch (error) {
    if (runtime.backend !== 'webgpu') throw error;
    abandonGpu(runtime, error);
    return scoreWith(await load(), image);
  }
}
