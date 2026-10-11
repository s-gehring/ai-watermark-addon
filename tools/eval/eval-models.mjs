import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
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
const CACHE = new URL('./.cache/', import.meta.url).pathname;
const SAMPLES = join(CACHE, 'samples');
const SCORES = join(CACHE, 'scores');
const bundled = JSON.parse(await readFile(join(root, 'model.json'), 'utf8'));
env.cacheDir = join(CACHE, 'models');

const CANDIDATES = [
  { id: 'bundled', label: `${bundled.name} [bundled ${bundled.dtype}]` },
  { id: 'buildborderless/CommunityForensics-DeepfakeDet-ViT', dtype: 'fp32', output: 'sigmoid' },
  { id: 'onnx-community/SMOGY-Ai-images-detector-ONNX', dtype: 'q8', output: { softmaxLabels: ['artificial'] } },
  { id: 'neilxxx/ai-vs-human-SigLIP-ONNX', dtype: 'q8', output: { softmaxLabels: ['ai'] } },
  { id: 'prithivMLmods/AI-vs-Deepfake-vs-Real-ONNX', dtype: 'q8', output: { softmaxLabels: ['Artificial', 'Deepfake'] } },
];

const CONDITIONS = {
  original: null,
  web1280: { size: 1280, quality: 80 },
  web800: { size: 800, quality: 75 },
};
const CALIBRATION_CONDITIONS = ['web1280', 'web800'];
const PHOTO_GROUPS = ['core', 'reddit', 'coco'];
const TARGET_FPR = 0.01;

const args = process.argv.slice(2);
const only = args.find((a) => !a.startsWith('--')) ?? 'bundled';
const calibrate = args.includes('--calibrate');
const perGenerator = args.includes('--per-generator');

async function listSamples() {
  const out = [];
  for (const group of await readdir(SAMPLES).catch(() => [])) {
    for (const label of ['fake', 'real']) {
      for (const file of await readdir(join(SAMPLES, group, label)).catch(() => [])) {
        const key = `${group}/${label}/${file}`;
        out.push({
          key,
          group,
          fake: label === 'fake',
          path: join(SAMPLES, key),
          source: file.split('__')[1]?.replace(/\.\w+$/, ''),
          split: createHash('sha1').update(key).digest()[0] & 1 ? 'test' : 'calibration',
        });
      }
    }
  }
  return out;
}

async function loadPixels(path, condition) {
  let input = await readFile(path);
  if (condition) {
    input = await sharp(input)
      .resize(condition.size, condition.size, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: condition.quality })
      .toBuffer();
  }
  const { data, info } = await sharp(input).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, channels: 3, data };
}

async function bundledScorer() {
  env.allowRemoteModels = false;
  env.localModelPath = join(root, 'vendor', 'models') + '/';
  const model = await AutoModelForImageClassification.from_pretrained(bundled.name, { dtype: bundled.dtype });
  const { crop } = bundled.input;
  return {
    model,
    async logit(image) {
      const pixels = toModelInput(image, bundled.input);
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
    async logit(image) {
      const { logits } = await model(await processor(new RawImage(image.data, image.width, image.height, 3)));
      return aiLogit(logits.data, c.output, model.config.id2label);
    },
  };
}

function cacheName(c) {
  const identity = c.id === 'bundled' ? { revision: bundled.revision, dtype: bundled.dtype, input: bundled.input } : c;
  const hash = createHash('sha1').update(JSON.stringify(identity)).digest('hex').slice(0, 10);
  return `${c.id.replace(/\W+/g, '_')}-${hash}.json`;
}

async function scoreAll(c, samples) {
  const file = join(SCORES, cacheName(c));
  const cached = JSON.parse(await readFile(file, 'utf8').catch(() => '{}'));
  const missing = samples.flatMap((s) => Object.keys(CONDITIONS).filter((k) => cached[`${s.key}|${k}`] === undefined).map((k) => [s, k]));
  let ms = null;
  if (missing.length) {
    const scorer = c.id === 'bundled' ? await bundledScorer() : await hubScorer(c);
    const t0 = performance.now();
    let done = 0;
    for (const [s, k] of missing) {
      try {
        cached[`${s.key}|${k}`] = await scorer.logit(await loadPixels(s.path, CONDITIONS[k]));
      } catch (e) {
        console.warn(`skip ${s.key} (${k}): ${e.message}`);
        cached[`${s.key}|${k}`] = null;
      }
      if (++done % 500 === 0) {
        console.log(`  scored ${done}/${missing.length}`);
        await mkdir(SCORES, { recursive: true });
        await writeFile(file, JSON.stringify(cached));
      }
    }
    ms = (performance.now() - t0) / missing.length;
    await mkdir(SCORES, { recursive: true });
    await writeFile(file, JSON.stringify(cached));
    await scorer.model.dispose();
  }
  const scored = [];
  for (const s of samples) {
    for (const condition of Object.keys(CONDITIONS)) {
      const z = cached[`${s.key}|${condition}`];
      if (z !== null) scored.push({ ...s, condition, z });
    }
  }
  return { scored, ms };
}

function auc(pos, neg) {
  const sorted = [...neg].sort((a, b) => a - b);
  let wins = 0;
  for (const p of pos) {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] < p) lo = mid + 1;
      else hi = mid;
    }
    let eq = lo;
    while (eq < sorted.length && sorted[eq] === p) eq++;
    wins += lo + (eq - lo) / 2;
  }
  return wins / (pos.length * neg.length);
}

function tprAtFpr(pos, neg, fpr) {
  const threshold = [...neg].sort((a, b) => b - a)[Math.floor(fpr * neg.length)];
  return pos.filter((p) => p > threshold).length / pos.length;
}

function wilson(k, n) {
  const z = 1.96;
  const p = k / n;
  const center = (p + (z * z) / (2 * n)) / (1 + (z * z) / n);
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / (1 + (z * z) / n);
  return [Math.max(0, center - half), Math.min(1, center + half)];
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
  return { scale: Number(a.toFixed(4)), bias: Number(b.toFixed(4)) };
}

function thresholdForFpr(realProbabilities, fpr) {
  const sorted = [...realProbabilities].sort((a, b) => b - a);
  return Math.min(0.99, (Math.floor(sorted[Math.floor(fpr * sorted.length)] * 100) + 1) / 100);
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;
const cell = (s, width = 20) => String(s).padEnd(width);

function report(test, threshold) {
  const conditions = Object.keys(CONDITIONS);
  const by = (filter) => Object.fromEntries(conditions.map((k) => [k, test.filter((s) => s.condition === k && filter(s))]));
  const line = (name, values) => console.log(cell(name, 30) + values.map((v) => cell(v)).join(''));
  line('', conditions);

  const photos = by((s) => PHOTO_GROUPS.includes(s.group));
  const zs = (rows, fake) => rows.filter((s) => s.fake === fake).map((s) => s.z);
  line('AUC (photos)', conditions.map((k) => auc(zs(photos[k], true), zs(photos[k], false)).toFixed(3)));
  for (const fpr of [0.005, 0.01, 0.02]) {
    line(`detected @ ${pct(fpr)} flagged`, conditions.map((k) => pct(tprAtFpr(zs(photos[k], true), zs(photos[k], false), fpr))));
  }
  for (const t of [...new Set([0.5, 0.7, 0.8, 0.9, threshold, 0.97])].sort()) {
    const at = (rows, fake) => rows.filter((s) => s.fake === fake && s.p >= t).length / rows.filter((s) => s.fake === fake).length;
    line(`detected / flagged @ ${t}`, conditions.map((k) => `${pct(at(photos[k], true))} / ${pct(at(photos[k], false))}`));
  }

  console.log(`at threshold ${threshold}:`);
  const rate = (rows) => {
    const k = rows.filter((s) => s.p >= threshold).length;
    const [lo, hi] = wilson(k, rows.length);
    return `${pct(k / rows.length)} (${(lo * 100).toFixed(1)}–${(hi * 100).toFixed(1)})`;
  };
  const fakeGroups = [...new Set(test.filter((s) => s.fake).map((s) => s.group))].sort();
  for (const g of fakeGroups) {
    const rows = by((s) => s.fake && s.group === g);
    line(`  detected ${g} (n=${rows.original.length})`, conditions.map((k) => rate(rows[k])));
  }
  const realSources = [...new Set(test.filter((s) => !s.fake).map((s) => `${s.group}/${s.source}`))].sort();
  for (const source of realSources) {
    const rows = by((s) => !s.fake && `${s.group}/${s.source}` === source);
    line(`  flagged ${source.split('/')[1]} (n=${rows.original.length})`, conditions.map((k) => rate(rows[k])));
  }
  line('  flagged photos (pooled)', conditions.map((k) => rate(photos[k].filter((s) => !s.fake))));

  if (perGenerator) {
    for (const g of fakeGroups) {
      const generators = [...new Set(test.filter((s) => s.fake && s.group === g).map((s) => s.source))].sort();
      for (const generator of generators) {
        const rows = by((s) => s.fake && s.group === g && s.source === generator);
        line(`    ${generator}`, conditions.map((k) => `${rows[k].filter((s) => s.p >= threshold).length}/${rows[k].length}`));
      }
    }
  }
}

const samples = await listSamples();
const count = (split, fake) => samples.filter((s) => s.split === split && s.fake === fake).length;
console.log(`samples: ${samples.length} (calibration ${count('calibration', true)} AI / ${count('calibration', false)} real, test ${count('test', true)} AI / ${count('test', false)} real)`);

for (const c of CANDIDATES) {
  if (only !== 'all' && c.id !== only && !c.id.includes(only)) continue;
  const { scored, ms } = await scoreAll(c, samples);
  const calibrationSet = scored.filter((s) => s.split === 'calibration' && PHOTO_GROUPS.includes(s.group) && CALIBRATION_CONDITIONS.includes(s.condition));

  let calibration = bundled.calibration;
  let threshold = bundled.threshold;
  if (calibrate || c.id !== 'bundled') {
    calibration = fitPlatt(calibrationSet);
    threshold = thresholdForFpr(calibrationSet.filter((s) => !s.fake).map((s) => calibratedProbability(s.z, calibration)), TARGET_FPR);
  }
  for (const s of scored) s.p = calibratedProbability(s.z, calibration);

  console.log(`\n${c.label ?? `${c.id} [${c.dtype}]`}${ms === null ? '' : ` | ${ms.toFixed(0)} ms/img`}`);
  console.log(`calibration ${JSON.stringify(calibration)} threshold ${threshold}${calibrate || c.id !== 'bundled' ? ` (fitted for ${pct(TARGET_FPR)} flagged photos on ${CALIBRATION_CONDITIONS.join(', ')})` : ' (model.json)'}`);
  report(scored.filter((s) => s.split === 'test'), threshold);
}
