import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const model = JSON.parse(await readFile(join(root, 'model.json'), 'utf8'));
const target = join(root, 'vendor', 'models', model.name);

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

for (const [file, expected] of Object.entries(model.files)) {
  const dest = join(target, file);
  const existing = await readFile(dest).catch(() => null);
  if (existing && (!expected || sha256(existing) === expected)) {
    console.log(`ok       ${file}`);
    continue;
  }
  const url = `https://huggingface.co/${model.repo}/resolve/${model.revision}/${file}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (expected && sha256(buf) !== expected) {
    throw new Error(`checksum mismatch for ${file}`);
  }
  await mkdir(dirname(dest), { recursive: true });
  await writeFile(dest, buf);
  console.log(`fetched  ${file} (${(buf.length / 1e6).toFixed(1)} MB)`);
}
