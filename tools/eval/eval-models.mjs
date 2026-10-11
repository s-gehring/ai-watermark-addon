import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import sharp from 'sharp';
import {
  AutoModelForImageClassification,
  AutoProcessor,
  RawImage,
  Tensor,
  env,
} from '@huggingface/transformers';
import { toModelInput } from '../../src/background/preprocess.ts';
import { aiLogit, calibratedProbability } from '../../src/background/score.ts';

const root = new URL('../../', import.meta.url).pathname;
const SAMPLES = new URL('./.cache/samples/', import.meta.url).pathname;
const bundled = JSON.parse(await readFile(join(root, 'model.json'), 'utf8'));
env.cacheDir = new URL('./.cache/models/', import.meta.url).pathname;

const CANDIDATES = [
  { id: 'bundled', label: `${bundled.name} [bundled ${bundled.dtype}]` },
  { id: 'buildborderless/CommunityForensics-DeepfakeDet-ViT', dtype: 'fp32', output: 'sigmoid' },
  { id: 'onnx-community/SMOGY-Ai-images-detector-ONNX', dtype: 'q8', output: { softmaxLabels: ['artificial'] } },
  { id: 'neilxxx/ai-vs-human-SigLIP-ONNX', dtype: 'q8', output: { softmaxLabels: ['ai'] } },
  { id: 'prithivMLmods/AI-vs-Deepfake-vs-Real-ONNX', dtype: 'q8', output: { softmaxLabels: ['Artificial', 'Deepfake'] } },
];

const args = process.argv.slice(2);
const only = args.find((a) => !a.startsWith('--')) ?? 'bundled';
const calibrate = args.includes('--calibrate');
const perGenerator = args.includes('--per-generator');

async function listSamples() {
  const out = [];
  for (const set of ['core', 'wild']) {
    for (const label of ['fake', 'real']) {
      const dir = join(SAMPLES, set, label);
      for (const f of await readdir(dir).catch(() => [])) {
        out.push({ set, fake: label === 'fake', path: join(dir, f), generator: f.split('__')[1]?.replace(/\.\w+$/, '') });
      }
    }
  }
  return out;
}

async function bundledScorer() {
  env.allowRemoteModels = false;
  env.localModelPath = join(root, 'vendor', 'models') + '/';
  const model = await AutoModelForImageClassification.from_pretrained(bundled.name, { dtype: bundled.dtype });
  const { crop } = bundled.input;
  return {
    model,
    async logit(path) {
      const { data, info } = await sharp(path).removeAlpha().raw().toBuffer({ resolveWithObject: true });
      const pixels = toModelInput({ width: info.width, height: info.height, channels: 3, data }, bundled.input);
      const { logits } = await model({ pixel_values: new Tensor('float32', pixels, [1, 3, crop, crop]) });
      return aiLogit(logits.data, bundled.output);
    },
  };
}

async function hubScorer(c) {
  env.allowRemoteModels = true;
  const processor = await AutoProcessor.from_pretrained(c.id);
  const model = await AutoModelForImageClassification.from_pretrained(c.id, { dtype: c.dtype });
  return {
    model,
    async logit(path) {
      const { logits } = await model(await processor((await RawImage.read(path)).rgb()));
      return aiLogit(logits.data, c.output, model.config.id2label);
    },
  };
}

function auc(scored) {
  const pos = scored.filter((s) => s.fake).map((s) => s.z);
  const neg = scored.filter((s) => !s.fake).map((s) => s.z);
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}

function rates(scored, threshold) {
  const pos = scored.filter((s) => s.fake);
  const neg = scored.filter((s) => !s.fake);
  return {
    tpr: pos.filter((s) => s.p >= threshold).length / (pos.length || 1),
    fpr: neg.filter((s) => s.p >= threshold).length / (neg.length || 1),
  };
}

function plattLoss(points, a, b, pos, neg) {
  let loss = 0;
  for (const { z, fake } of points) {
    const t = a * z + b;
    loss += (fake ? 0.5 / pos : 0.5 / neg) * (fake ? Math.log1p(Math.exp(-t)) : Math.log1p(Math.exp(t)));
  }
  return loss;
}

function fitPlatt(points) {
  const pos = points.filter((p) => p.fake).length;
  const neg = points.length - pos;
  let a = 1;
  let b = 0;
  let loss = plattLoss(points, a, b, pos, neg);
  for (let iter = 0; iter < 200; iter++) {
    let ga = 0, gb = 0, haa = 1e-9, hab = 0, hbb = 1e-9;
    for (const { z, fake } of points) {
      const w = fake ? 0.5 / pos : 0.5 / neg;
      const p = 1 / (1 + Math.exp(-(a * z + b)));
      const r = w * (p - (fake ? 1 : 0));
      const h = w * p * (1 - p);
      ga += r * z; gb += r; haa += h * z * z; hab += h * z; hbb += h;
    }
    const det = haa * hbb - hab * hab;
    const da = (hbb * ga - hab * gb) / det;
    const db = (haa * gb - hab * ga) / det;
    let step = 1;
    while (step > 1e-6 && plattLoss(points, a - step * da, b - step * db, pos, neg) > loss) step /= 2;
    a -= step * da;
    b -= step * db;
    const next = plattLoss(points, a, b, pos, neg);
    if (loss - next < 1e-12) break;
    loss = next;
  }
  return { scale: a, bias: b };
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;
const samples = await listSamples();
console.log(`samples: ${samples.length}`);

for (const c of CANDIDATES) {
  if (only !== 'all' && c.id !== only && !c.id.includes(only)) continue;
  const t0 = performance.now();
  const scorer = c.id === 'bundled' ? await bundledScorer() : await hubScorer(c);
  const scored = [];
  for (const s of samples) {
    try {
      scored.push({ ...s, z: await scorer.logit(s.path) });
    } catch (e) {
      console.warn(`skip ${s.path}: ${e.message}`);
    }
  }
  const ms = (performance.now() - t0) / scored.length;

  let calibration = c.id === 'bundled' ? bundled.calibration : { scale: 1, bias: 0 };
  const core = scored.filter((s) => s.set === 'core');
  if (calibrate) calibration = fitPlatt(core);
  for (const s of scored) s.p = calibratedProbability(s.z, calibration);
  const wildFake = scored.filter((s) => s.set === 'wild' && s.fake);

  console.log(`${c.label ?? `${c.id} [${c.dtype}]`} | AUC ${auc(core).toFixed(3)} | ${ms.toFixed(0)} ms/img`);
  console.log(`  calibration: scale ${calibration.scale.toFixed(4)} bias ${calibration.bias.toFixed(4)}`);
  for (const t of [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]) {
    const r = rates(core, t);
    console.log(`  thr ${t}: TPR ${pct(r.tpr)} FPR ${pct(r.fpr)} | wild TPR ${pct(rates(wildFake, t).tpr)}`);
  }
  if (perGenerator) {
    const threshold = c.id === 'bundled' ? bundled.threshold : 0.5;
    const byGen = new Map();
    for (const s of core.filter((x) => x.fake)) {
      byGen.set(s.generator, [...(byGen.get(s.generator) ?? []), s.p >= threshold]);
    }
    for (const [g, hits] of [...byGen].sort()) {
      console.log(`    ${g}: ${hits.filter(Boolean).length}/${hits.length}`);
    }
  }
  await scorer.model.dispose();
}
