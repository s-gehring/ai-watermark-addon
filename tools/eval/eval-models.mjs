import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import sharp from 'sharp';
import * as ort from 'onnxruntime-node';
import {
  AutoModelForImageClassification,
  AutoProcessor,
  RawImage,
  env,
} from '@huggingface/transformers';
import { inputShape, toModelInput } from '../../src/background/preprocess.ts';
import { aiLogit, calibratedProbability } from '../../src/background/score.ts';

const root = new URL('../../', import.meta.url).pathname;

function mainCheckout() {
  try {
    return dirname(execFileSync('git', ['rev-parse', '--path-format=absolute', '--git-common-dir'], { cwd: root, encoding: 'utf8' }).trim());
  } catch {
    return root;
  }
}

const CACHE = join(mainCheckout(), 'tools', 'eval', '.cache');
const SAMPLES = join(CACHE, 'samples');
const SCORES = join(CACHE, 'scores');
const config = JSON.parse(await readFile(join(root, 'model.json'), 'utf8'));
env.cacheDir = join(CACHE, 'models');

const BUNDLED = config.models.map((m) => ({ ...m, id: m.name, local: true }));
const COMMUNITY_FORENSICS_INPUT = {
  shortestEdge: 440,
  crop: 384,
  resample: 'bicubic',
  mean: [0.48145466, 0.4578275, 0.40821073],
  std: [0.26862954, 0.26130258, 0.27577711],
};
const HUB = [
  { id: 'buildborderless/CommunityForensics-DeepfakeDet-ViT', dtype: 'fp32', output: 'sigmoid' },
  { id: 'neilxxx/ai-vs-human-SigLIP-ONNX', dtype: 'q8', output: { softmaxLabels: ['ai'] } },
  { id: 'onnx-community/SMOGY-Ai-images-detector-ONNX', dtype: 'q8', output: { softmaxLabels: ['artificial'] } },
  { id: 'prithivMLmods/AI-vs-Deepfake-vs-Real-ONNX', dtype: 'q8', output: { softmaxLabels: ['Artificial', 'Deepfake'] } },
  { id: 'onnx-community/ai-image-detection-ONNX', dtype: 'q8', output: { softmaxLabels: ['FAKE'] } },
  { id: 'onnx-community/ai-image-detect-distilled-ONNX', dtype: 'q8', output: { softmaxLabels: ['fake'] } },
  {
    id: 'angelhd25/ull-ai-image-detector',
    revision: '88ae192e4001bcbbf6f02de75e033171e227b5b2',
    file: 'commfor384_web_v13_fp32.onnx',
    dtype: 'fp32',
    output: 'sigmoid',
    input: COMMUNITY_FORENSICS_INPUT,
  },
];
const PRECOMPUTED = (await readdir(join(SCORES, 'precomputed')).catch(() => []))
  .filter((file) => file.endsWith('.json'))
  .map((file) => ({ id: `precomputed/${file.slice(0, -'.json'.length)}`, dtype: 'precomputed', precomputed: true }));
const args = process.argv.slice(2);
const only = args.find((a) => !a.startsWith('--')) ?? 'bundled';
const calibrate = args.includes('--calibrate');
const perGenerator = args.includes('--per-generator');
const membersArg = args.find((a) => a.startsWith('--members='))?.slice('--members='.length);

function findMember(id) {
  const member = [...BUNDLED, ...HUB, ...PRECOMPUTED].find((m) => m.id === id);
  if (!member) throw new Error(`unknown model ${id}`);
  return member;
}

const CANDIDATES = [
  { id: 'bundled', label: `bundled: ${BUNDLED.map((m) => `${m.name} [${m.dtype}]`).join(' + ')}`, members: BUNDLED, config },
  ...(BUNDLED.length > 1 ? BUNDLED.map((m) => ({ id: `bundled/${m.name}`, label: `${m.name} [${m.dtype}] alone`, members: [m] })) : []),
  ...HUB.map((h) => ({ id: h.id, label: `${h.id} [${h.dtype}]`, members: [h] })),
  ...HUB.map((h) => ({ id: `bundled+${h.id}`, label: `bundled + ${h.id} [${h.dtype}]`, members: [...BUNDLED, h] })),
  ...(membersArg ? [{ id: 'members', label: membersArg.replaceAll(',', ' + '), members: membersArg.split(',').map(findMember) }] : []),
];

const CONDITIONS = {
  original: null,
  web1280: { size: 1280, quality: 80 },
  web800: { size: 800, quality: 75 },
};
const CALIBRATION_CONDITIONS = ['web1280', 'web800'];
const PHOTO_GROUPS = ['core', 'reddit', 'coco', 'youtube'];
const TARGET_FPR = 0.01;
const ART_LIMIT = 0.05;

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

async function sessionScorer(path, m, id2label) {
  const session = await ort.InferenceSession.create(path, {
    intraOpNumThreads: Number(process.env.EVAL_THREADS) || 0,
    extra: { session: { intra_op: { allow_spinning: '0' } } },
  });
  const [width, height] = inputShape(m.input);
  return {
    model: { dispose: () => session.release() },
    async logit(image) {
      const input = new ort.Tensor('float32', toModelInput(image, m.input), [1, 3, height, width]);
      const output = await session.run({ [session.inputNames[0]]: input });
      return aiLogit(output[session.outputNames[0]].data, m.output, id2label);
    },
  };
}

async function localScorer(m) {
  const dir = join(root, 'vendor', 'models', m.name);
  const files = Object.keys(m.files);
  const modelConfig = files.includes('config.json') ? JSON.parse(await readFile(join(dir, 'config.json'), 'utf8')) : {};
  return sessionScorer(join(dir, files.find((file) => file.endsWith('.onnx'))), m, modelConfig.id2label);
}

async function hubScorer(m) {
  env.allowRemoteModels = true;
  const processor = await AutoProcessor.from_pretrained(m.id);
  const model = await AutoModelForImageClassification.from_pretrained(m.id, { dtype: m.dtype });
  return {
    model,
    async logit(image) {
      const { logits } = await model(await processor(new RawImage(image.data, image.width, image.height, 3)));
      return aiLogit(logits.data, m.output, model.config.id2label);
    },
  };
}

async function onnxScorer(m) {
  const path = join(CACHE, 'models', m.id, m.revision, m.file);
  if (!(await stat(path).then(() => true, () => false))) {
    const res = await fetch(`https://huggingface.co/${m.id}/resolve/${m.revision}/${m.file}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${m.id}/${m.file}`);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, Buffer.from(await res.arrayBuffer()));
  }
  return sessionScorer(path, m);
}

function cacheName(m) {
  if (m.precomputed) return `${m.id}.json`;
  const identity = m.local
    ? { repo: m.repo, revision: m.revision, dtype: m.dtype, input: m.input, output: m.output, ...(m.repo ? {} : { files: m.files }) }
    : m;
  const hash = createHash('sha1').update(JSON.stringify(identity)).digest('hex').slice(0, 10);
  return `${m.id.replace(/\W+/g, '_')}-${hash}.json`;
}

async function scoreMember(m, samples) {
  const file = join(SCORES, cacheName(m));
  const cached = JSON.parse(await readFile(file, 'utf8').catch(() => '{}'));
  const missing = samples.flatMap((s) => Object.keys(CONDITIONS).filter((k) => cached[`${s.key}|${k}`] === undefined).map((k) => [s, k]));
  if (!missing.length) return { cached, ms: null };
  if (m.precomputed) throw new Error(`${m.id} lacks ${missing.length} scores`);
  const scorer = m.local ? await localScorer(m) : m.file ? await onnxScorer(m) : await hubScorer(m);
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
      console.log(`  ${m.id}: scored ${done}/${missing.length}`);
      await mkdir(SCORES, { recursive: true });
      await writeFile(file, JSON.stringify(cached));
    }
  }
  await mkdir(SCORES, { recursive: true });
  await writeFile(file, JSON.stringify(cached));
  await scorer.model.dispose();
  return { cached, ms: (performance.now() - t0) / missing.length };
}

async function scoreAll(c, samples) {
  const results = [];
  for (const m of c.members) results.push(await scoreMember(m, samples));
  const scored = [];
  for (const s of samples) {
    for (const condition of Object.keys(CONDITIONS)) {
      const zs = results.map((r) => r.cached[`${s.key}|${condition}`]);
      if (!zs.includes(null)) scored.push({ ...s, condition, zs });
    }
  }
  const timed = results.filter((r) => r.ms !== null);
  return { scored, ms: timed.length === results.length ? timed.reduce((sum, r) => sum + r.ms, 0) : null };
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

function logisticLoss(points, theta, weightOf) {
  let loss = 0;
  for (const { zs, fake } of points) {
    const t = zs.reduce((sum, z, i) => sum + theta[i] * z, theta[zs.length]);
    loss += weightOf(fake) * (fake ? Math.log1p(Math.exp(-t)) : Math.log1p(Math.exp(t)));
  }
  return loss;
}

function solve(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col++) {
    const pivot = a.slice(col).reduce((best, row, i) => (Math.abs(row[col]) > Math.abs(a[best][col]) ? col + i : best), col);
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let row = col + 1; row < n; row++) {
      const f = a[row][col] / a[col][col];
      for (let k = col; k <= n; k++) a[row][k] -= f * a[col][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    x[row] = (a[row][n] - a[row].slice(row + 1, n).reduce((sum, v, i) => sum + v * x[row + 1 + i], 0)) / a[row][row];
  }
  return x;
}

function fitLogistic(points) {
  const k = points[0].zs.length;
  const pos = points.filter((p) => p.fake).length;
  const weightOf = (fake) => (fake ? 0.5 / pos : 0.5 / (points.length - pos));
  let theta = new Array(k + 1).fill(0);
  let loss = logisticLoss(points, theta, weightOf);
  for (let iter = 0; iter < 200; iter++) {
    const grad = new Array(k + 1).fill(0);
    const hess = Array.from({ length: k + 1 }, (_, i) => Array.from({ length: k + 1 }, (_, j) => (i === j ? 1e-9 : 0)));
    for (const { zs, fake } of points) {
      const x = [...zs, 1];
      const p = 1 / (1 + Math.exp(-x.reduce((sum, v, i) => sum + theta[i] * v, 0)));
      const w = weightOf(fake);
      for (let i = 0; i <= k; i++) {
        grad[i] += w * (p - (fake ? 1 : 0)) * x[i];
        for (let j = 0; j <= k; j++) hess[i][j] += w * p * (1 - p) * x[i] * x[j];
      }
    }
    const delta = solve(hess, grad);
    let step = 1;
    const at = (st) => theta.map((t, i) => t - st * delta[i]);
    while (step > 1e-6 && logisticLoss(points, at(step), weightOf) > loss) step /= 2;
    theta = at(step);
    const next = logisticLoss(points, theta, weightOf);
    if (loss - next < 1e-12) break;
    loss = next;
  }
  return { weights: theta.slice(0, k).map((w) => Number(w.toFixed(4))), bias: Number(theta[k].toFixed(4)) };
}

function rankThreshold(realScores, fpr) {
  return [...realScores].sort((a, b) => b - a)[Math.floor(fpr * realScores.length)];
}

function fitEnsemble(calibrationSet) {
  const k = calibrationSet[0].zs.length;
  const withoutArt = calibrationSet.filter((s) => s.group !== 'art');
  if (k === 1) return fitLogistic(withoutArt);
  const combine = (ratios) => (s) => s.zs.reduce((sum, z, i) => sum + ratios[i] * z, 0);
  const detected = (ratios) => {
    const combined = combine(ratios);
    const threshold = rankThreshold(withoutArt.filter((s) => !s.fake).map(combined), TARGET_FPR);
    const share = (rows) => rows.filter((s) => combined(s) > threshold).length / rows.length;
    return share(calibrationSet.filter((s) => s.group === 'art')) <= ART_LIMIT ? share(calibrationSet.filter((s) => s.fake)) : -1;
  };
  const ratios = [1, ...new Array(k - 1).fill(0)];
  let best = detected(ratios);
  for (let improved = true; improved; ) {
    improved = false;
    for (let i = 1; i < k; i++) {
      for (let step = 0; step <= 100; step++) {
        const candidate = ratios.with(i, step / 20);
        const score = detected(candidate);
        if (score > best) {
          best = score;
          ratios[i] = step / 20;
          improved = true;
        }
      }
    }
  }
  const platt = fitLogistic(withoutArt.map((s) => ({ ...s, zs: [combine(ratios)(s)] })));
  const [scale] = platt.weights;
  return { weights: ratios.map((r) => Number((scale * r).toFixed(4))), bias: platt.bias };
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

  const photos = by(isRelevant);
  const zs = (rows, fake) => rows.filter((s) => s.fake === fake).map((s) => s.p);
  line('AUC (AI vs photos)', conditions.map((k) => auc(zs(photos[k], true), zs(photos[k], false)).toFixed(3)));
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

const isRelevant = (s) => s.fake || PHOTO_GROUPS.includes(s.group);
const samples = await listSamples();
const count = (split, fake) => samples.filter((s) => s.split === split && s.fake === fake).length;
console.log(`samples: ${samples.length} (calibration ${count('calibration', true)} AI / ${count('calibration', false)} real, test ${count('test', true)} AI / ${count('test', false)} real)`);

for (const c of CANDIDATES) {
  if (only !== 'all' && c.id !== only && !c.id.startsWith(`${only}/`) && (only === 'bundled' || !c.id.includes(only))) continue;
  const { scored, ms } = await scoreAll(c, samples);
  const calibrationSet = scored.filter((s) => s.split === 'calibration' && (isRelevant(s) || s.group === 'art') && CALIBRATION_CONDITIONS.includes(s.condition));

  const fitted = calibrate || !c.config;
  const calibration = fitted ? fitEnsemble(calibrationSet) : c.config.calibration;
  for (const s of scored) s.p = calibratedProbability(s.zs, calibration);
  const photos = calibrationSet.filter((s) => !s.fake && s.group !== 'art');
  const threshold = fitted ? thresholdForFpr(photos.map((s) => s.p), TARGET_FPR) : c.config.threshold;

  console.log(`\n${c.label}${ms === null ? '' : ` | ${ms.toFixed(0)} ms/img`}`);
  const target = `${pct(TARGET_FPR)} flagged photos${c.members.length > 1 ? ` and at most ${pct(ART_LIMIT)} flagged paintings` : ''} on ${CALIBRATION_CONDITIONS.join(', ')}`;
  console.log(`calibration ${JSON.stringify(calibration)} threshold ${threshold}${fitted ? ` (fitted for ${target})` : ' (model.json)'}`);
  report(scored.filter((s) => s.split === 'test'), threshold);
}
