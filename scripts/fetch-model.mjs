import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const { models } = JSON.parse(await readFile(join(root, 'model.json'), 'utf8'));

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

for (const model of models) {
  for (const [file, expected] of Object.entries(model.files)) {
    const dest = join(root, 'vendor', 'models', model.name, file);
    const existing = await readFile(dest).catch(() => null);
    if (existing && (!expected || sha256(existing) === expected)) {
      console.log(`ok       ${model.name}/${file}`);
      continue;
    }
    if (!model.repo) {
      throw new Error(`${model.name}/${file} has no download source; place it in vendor/models/${model.name}/`);
    }
    const url = `https://huggingface.co/${model.repo}/resolve/${model.revision}/${file}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${url}`);
    const buf = Buffer.from(await res.arrayBuffer());
    if (expected && sha256(buf) !== expected) {
      throw new Error(`checksum mismatch for ${model.name}/${file}`);
    }
    await mkdir(dirname(dest), { recursive: true });
    await writeFile(dest, buf);
    console.log(`fetched  ${model.name}/${file} (${(buf.length / 1e6).toFixed(1)} MB)`);
  }
}
