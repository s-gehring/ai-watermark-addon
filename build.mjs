import { build, context } from 'esbuild';
import { cp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const root = new URL('.', import.meta.url).pathname;
const out = join(root, 'dist');
const watch = process.argv.includes('--watch');
const pkg = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));
const model = JSON.parse(await readFile(join(root, 'model.json'), 'utf8'));
const modelDir = join(root, 'vendor', 'models', model.name);

const exists = (path) => stat(path).then(() => true, () => false);

if (!(await exists(join(modelDir, Object.keys(model.files).at(-1))))) {
  console.error('Model files missing. Run `npm run fetch-model` first.');
  process.exit(1);
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
  { ...common, entryPoints: { background: 'src/background/index.ts' }, format: 'esm', outdir: out },
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
  ['src/popup/popup.html', 'popup.html'],
  ['src/popup/popup.css', 'popup.css'],
  ['src/options/options.html', 'options.html'],
  ['src/options/options.css', 'options.css'],
  ['node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs', 'ort/ort-wasm-simd-threaded.asyncify.mjs'],
  ['node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm', 'ort/ort-wasm-simd-threaded.asyncify.wasm'],
];
for (const [from, to] of copies) {
  await cp(join(root, from), join(out, to), { recursive: true });
}
for (const file of Object.keys(model.files)) {
  await cp(join(modelDir, file), join(out, 'models', model.name, file));
}

if (watch) {
  for (const config of configs) await (await context(config)).watch();
} else {
  await Promise.all(configs.map((config) => build(config)));
}
