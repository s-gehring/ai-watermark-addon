import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import config from '../../model.json';
import type { PixelImage } from '../../src/background/preprocess';

const SELF_TEST_LOGIT = config.models[0].selfTestLogit;
const GPU_LOGIT = 3;
const WASM_LOGIT = -3;

interface FakeOrt {
  created: number;
  runs: number;
  released: number;
  createError: Error | null;
  createHangs: boolean;
  selfTestLogit: number;
  runError: Error | null;
  lose: (info: { message: string }) => void;
  env: { wasm: Record<string, unknown>; webgpu: { device?: Promise<{ lost: Promise<{ message: string }> }> } };
  InferenceSession: { create: () => Promise<unknown> };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => { data: Float32Array };
}

const isSelfTest = (data: Float32Array) => data.length > 1 && data[1] === Math.fround(Math.sin(12.9898) * 2);

function fakeOrt(logit: number): FakeOrt {
  const ort: FakeOrt = {
    created: 0,
    runs: 0,
    released: 0,
    createError: null,
    createHangs: false,
    selfTestLogit: SELF_TEST_LOGIT,
    runError: null,
    lose: () => {},
    env: { wasm: {}, webgpu: {} },
    InferenceSession: {
      create: async () => {
        ort.created++;
        if (ort.createHangs) return new Promise(() => {});
        if (ort.createError) throw ort.createError;
        const lost = new Promise<{ message: string }>((resolve) => (ort.lose = resolve));
        ort.env.webgpu.device = Promise.resolve({ lost });
        return {
          inputNames: ['pixel_values'],
          outputNames: ['logits'],
          run: async ({ pixel_values: input }: { pixel_values: { data: Float32Array } }) => {
            ort.runs++;
            if (ort.runError) throw ort.runError;
            return { logits: { data: new Float32Array([isSelfTest(input.data) ? ort.selfTestLogit : logit]) } };
          },
          release: async () => {
            ort.released++;
          },
        };
      },
    },
    Tensor: class {
      constructor(
        readonly type: string,
        readonly data: Float32Array,
        readonly dims: number[],
      ) {}
    },
  };
  return ort;
}

const fakes = vi.hoisted(() => {
  const current = { wasm: null as unknown as FakeOrt, gpu: null as unknown as FakeOrt };
  const module = (name: 'wasm' | 'gpu') => ({
    get env() {
      return current[name].env;
    },
    get InferenceSession() {
      return current[name].InferenceSession;
    },
    get Tensor() {
      return current[name].Tensor;
    },
  });
  return { current, module };
});
vi.mock('onnxruntime-web/wasm', () => fakes.module('wasm'));
vi.mock('onnxruntime-web/jspi', () => fakes.module('gpu'));

const image: PixelImage = { width: 400, height: 400, channels: 4, data: new Uint8ClampedArray(400 * 400 * 4) };
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const scoreOf = (logit: number) => sigmoid(config.calibration.bias + config.calibration.weights[0] * logit);

let wasm: FakeOrt;
let gpu: FakeOrt;
let classifier: typeof import('../../src/background/classifier');

async function setup({ hasGpu = true, hasJspi = true, url = 'moz-extension://x/worker.js' } = {}) {
  vi.stubGlobal('self', { location: { href: url } });
  vi.stubGlobal('navigator', hasGpu ? { gpu: {} } : {});
  vi.stubGlobal('WebAssembly', hasJspi ? { Suspending: class {} } : {});
  classifier = await import('../../src/background/classifier');
}

beforeEach(() => {
  vi.resetModules();
  wasm = fakeOrt(WASM_LOGIT);
  gpu = fakeOrt(GPU_LOGIT);
  fakes.current.wasm = wasm;
  fakes.current.gpu = gpu;
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('backend selection', () => {
  it('runs on WebGPU when the self-test matches', async () => {
    await setup();
    const { score } = await classifier.scoreImage(image);
    expect(score).toBeCloseTo(scoreOf(GPU_LOGIT));
    expect(classifier.activeBackend()).toBe('webgpu');
    expect(wasm.created).toBe(0);
    expect(gpu.env.wasm.wasmPaths).toEqual({ wasm: 'moz-extension://x/ort/ort-wasm-simd-threaded.jspi.wasm' });
  });

  it.each([
    ['WebGPU is missing', { hasGpu: false }],
    ['JSPI is missing', { hasJspi: false }],
    ['the worker was started for WASM', { url: 'moz-extension://x/worker.js?backend=wasm' }],
  ])('stays on WASM when %s', async (_, options) => {
    await setup(options);
    const { score } = await classifier.scoreImage(image);
    expect(score).toBeCloseTo(scoreOf(WASM_LOGIT));
    expect(classifier.activeBackend()).toBe('wasm');
    expect(gpu.created).toBe(0);
    expect(wasm.env.wasm.wasmPaths).toEqual({ wasm: 'moz-extension://x/ort/ort-wasm-simd-threaded.wasm' });
  });

  it('falls back to WASM when the WebGPU session cannot be created', async () => {
    gpu.createError = new Error('no adapter');
    await setup();
    expect((await classifier.scoreImage(image)).score).toBeCloseTo(scoreOf(WASM_LOGIT));
    expect(classifier.activeBackend()).toBe('wasm');
  });

  it('falls back to WASM and releases the WebGPU session when the self-test deviates', async () => {
    gpu.selfTestLogit = SELF_TEST_LOGIT + 0.02;
    await setup();
    expect((await classifier.scoreImage(image)).score).toBeCloseTo(scoreOf(WASM_LOGIT));
    expect(classifier.activeBackend()).toBe('wasm');
    expect(gpu.released).toBe(1);
  });

  it('falls back to WASM when the self-test returns NaN', async () => {
    gpu.selfTestLogit = Number.NaN;
    await setup();
    expect((await classifier.scoreImage(image)).score).toBeCloseTo(scoreOf(WASM_LOGIT));
  });

  it('falls back to WASM when WebGPU does not start in time', async () => {
    vi.useFakeTimers();
    gpu.createHangs = true;
    await setup();
    const scored = classifier.scoreImage(image);
    await vi.advanceTimersByTimeAsync(classifier.GPU_INIT_TIMEOUT_MS);
    expect((await scored).score).toBeCloseTo(scoreOf(WASM_LOGIT));
    expect(classifier.activeBackend()).toBe('wasm');
  });
});

describe('fallback while running', () => {
  it('rescores the image on WASM and stays there when a WebGPU run fails', async () => {
    await setup();
    await classifier.scoreImage(image);
    gpu.runError = new Error('mapAsync failed');
    expect((await classifier.scoreImage(image)).score).toBeCloseTo(scoreOf(WASM_LOGIT));
    gpu.runError = null;
    const gpuRuns = gpu.runs;
    expect((await classifier.scoreImage(image)).score).toBeCloseTo(scoreOf(WASM_LOGIT));
    expect(gpu.runs).toBe(gpuRuns);
    expect(gpu.released).toBe(1);
    expect(classifier.activeBackend()).toBe('wasm');
  });

  it('switches to WASM after the GPU device is lost', async () => {
    await setup();
    await classifier.scoreImage(image);
    gpu.lose({ message: 'driver reset' });
    await vi.waitFor(() => expect(classifier.activeBackend()).toBeNull());
    expect((await classifier.scoreImage(image)).score).toBeCloseTo(scoreOf(WASM_LOGIT));
    expect(classifier.activeBackend()).toBe('wasm');
  });

  it('starts on WASM when the GPU device is lost during the self-test', async () => {
    await setup();
    const originalCreate = gpu.InferenceSession.create;
    gpu.InferenceSession.create = async () => {
      const session = await originalCreate();
      queueMicrotask(() => gpu.lose({ message: 'driver reset' }));
      return session;
    };
    expect((await classifier.scoreImage(image)).score).toBeCloseTo(scoreOf(WASM_LOGIT));
    expect(classifier.activeBackend()).toBe('wasm');
  });

  it('passes WASM errors through', async () => {
    await setup({ hasGpu: false });
    wasm.runError = new Error('out of memory');
    await expect(classifier.scoreImage(image)).rejects.toThrow('out of memory');
  });
});

