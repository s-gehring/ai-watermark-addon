import { build, context } from 'esbuild';
import { cp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const root = new URL('.', import.meta.url).pathname;
const out = join(root, 'dist');
const watch = process.argv.includes('--watch');
const pkg = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));
const { models } = JSON.parse(await readFile(join(root, 'model.json'), 'utf8'));
const modelFiles = models.flatMap((model) => Object.keys(model.files).map((file) => join(model.name, file)));

const exists = (path) => stat(path).then(() => true, () => false);

for (const file of modelFiles) {
  if (!(await exists(join(root, 'vendor', 'models', file)))) {
    console.error('Model files missing. Run `npm run fetch-model` first.');
    process.exit(1);
  }
}

await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });

const common = {
  bundle: true,
  target: 'firefox140',
  sourcemap: watch ? 'inline' : false,
  minify: !watch,
  legalComments: 'linked',
  logLevel: 'info',
};
const configs = [
  {
    ...common,
    entryPoints: { background: 'src/background/index.ts', worker: 'src/background/worker.ts' },
    format: 'esm',
    outdir: out,
  },
  {
    ...common,
    entryPoints: {
      content: 'src/content/index.ts',
      popup: 'src/popup/popup.ts',
      options: 'src/options/options.ts',
    },
    format: 'iife',
    outdir: out,
  },
];

const manifest = JSON.parse(await readFile(join(root, 'src', 'manifest.json'), 'utf8'));
manifest.version = pkg.version;
await writeFile(join(out, 'manifest.json'), JSON.stringify(manifest, null, 2));

const copies = [
  ['src/_locales', '_locales'],
  ['src/icons', 'icons'],
  ['src/shared/ui.css', 'ui.css'],
  ['src/background/synthid-templates', 'synthid-templates'],
  ['src/popup/popup.html', 'popup.html'],
  ['src/popup/popup.css', 'popup.css'],
  ['src/options/options.html', 'options.html'],
  ['src/options/options.css', 'options.css'],
  ['node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs', 'ort/ort-wasm-simd-threaded.mjs'],
  ['node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm', 'ort/ort-wasm-simd-threaded.wasm'],
];
for (const [from, to] of copies) {
  await cp(join(root, from), join(out, to), { recursive: true });
}
for (const file of modelFiles) {
  await cp(join(root, 'vendor', 'models', file), join(out, 'models', file));
}

if (watch) {
  for (const config of configs) await (await context(config)).watch();
} else {
  await Promise.all(configs.map((config) => build(config)));
}
