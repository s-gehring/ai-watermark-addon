import { createReadStream } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join } from 'node:path';
import { download } from 'geckodriver';
import { Builder } from 'selenium-webdriver';
import firefox from 'selenium-webdriver/firefox.js';

const root = new URL('../../', import.meta.url).pathname;
const ADDON_ID = 'ai-watermark-addon@s-gehring.github.io';
const ADDON_UUID = '6f0c3a52-9a51-4c1e-8f7e-2f6f5d1b7a11';
const TYPES = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' };

const config = JSON.parse(await readFile(join(root, 'model.json'), 'utf8'));
const thresholds = [config.cascade.band[0], config.possibleThreshold, config.threshold];
const weight = config.calibration.weights[0];
const limit = Number(process.env.LIMIT ?? Infinity);

async function imagesIn(dir) {
  const entries = await readdir(dir, { withFileTypes: true, recursive: true });
  return entries
    .filter((e) => e.isFile() && TYPES[extname(e.name).toLowerCase()])
    .map((e) => join(e.parentPath, e.name))
    .sort();
}

const dirs = process.argv.slice(2);
if (!dirs.length) {
  console.error('usage: node tools/eval/webgpu-parity.mjs <image dir>... (needs `npm run build`, runs a visible Firefox)');
  process.exit(2);
}
const files = (await Promise.all(dirs.map(imagesIn))).flat().slice(0, limit);
const server = createServer((req, res) => {
  const file = files[Number(req.url.slice(1))];
  res.writeHead(200, { 'content-type': TYPES[extname(file).toLowerCase()], 'access-control-allow-origin': '*' });
  createReadStream(file).pipe(res);
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const WORKERS = ['worker.js', 'worker.js?backend=wasm'];

const options = new firefox.Options().setPreference('extensions.webextensions.uuids', JSON.stringify({ [ADDON_ID]: ADDON_UUID }));
const driver = await new Builder()
  .forBrowser('firefox')
  .setFirefoxOptions(options)
  .setFirefoxService(new firefox.ServiceBuilder(await download()).addArguments('--allow-system-access'))
  .build();

try {
  await driver.installAddon(join(root, 'dist'), true);
  await driver.get(`moz-extension://${ADDON_UUID}/options.html`);
  await driver.manage().setTimeouts({ script: 600_000 });
  const results = [];
  await driver.executeScript((urls) => {
    window.parityWorkers = Object.fromEntries(urls.map((url) => [url, new Worker(url, { type: 'module' })]));
  }, WORKERS);
  for (const [index, file] of files.entries()) {
    const row = { file };
    for (const url of WORKERS) {
      const answer = await driver.executeAsyncScript(
        async (url, image, done) => {
          const res = await fetch(image);
          const buffer = await res.arrayBuffer();
          const worker = window.parityWorkers[url];
          const id = Math.random();
          const started = performance.now();
          worker.onmessage = ({ data }) => data.id === id && done({ ...data, ms: performance.now() - started });
          worker.postMessage({ id, bytes: buffer, type: res.headers.get('content-type'), original: null, checkMetadata: false, checkWatermark: false }, [buffer]);
        },
        url,
        `${origin}/${index}`,
      );
      row[answer.backend ?? 'none'] = answer;
    }
    results.push(row);
  }

  const logOdds = (p) => Math.log(p / (1 - p));
  const usable = results.filter((r) => r.webgpu?.verdict.ok && r.wasm?.verdict.ok);
  const missing = results.filter((r) => !r.webgpu || !r.wasm);
  const deltas = usable.map((r) => Math.abs(logOdds(r.webgpu.verdict.score) - logOdds(r.wasm.verdict.score)) / weight);
  const flips = thresholds.map((t) => usable.filter((r) => r.webgpu.verdict.score >= t !== r.wasm.verdict.score >= t).map((r) => r.file));
  const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
  console.log(
    JSON.stringify(
      {
        images: results.length,
        compared: usable.length,
        missingBackend: missing.map((r) => r.file),
        maxLogitDelta: Math.max(0, ...deltas),
        meanLogitDelta: deltas.reduce((a, b) => a + b, 0) / (deltas.length || 1),
        flips: Object.fromEntries(thresholds.map((t, i) => [t, flips[i]])),
        medianMs: { webgpu: median(usable.map((r) => r.webgpu.ms)), wasm: median(usable.map((r) => r.wasm.ms)) },
      },
      null,
      2,
    ),
  );
  if (missing.length || flips.some((f) => f.length)) process.exitCode = 1;
} finally {
  await driver.quit();
  server.close();
}
