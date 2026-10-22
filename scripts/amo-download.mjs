import { createHash, createHmac, randomUUID } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const PENDING = 3;
const DISABLED = 4;

const root = new URL('..', import.meta.url).pathname;
const [version, dest] = process.argv.slice(2);
const { AMO_JWT_ISSUER: issuer, AMO_JWT_SECRET: secret } = process.env;
if (!version || !dest || !issuer || !secret) {
  console.error('Usage: AMO_JWT_ISSUER=… AMO_JWT_SECRET=… node scripts/amo-download.mjs <version> <dest.xpi>');
  process.exit(1);
}

const manifest = JSON.parse(await readFile(join(root, 'src', 'manifest.json'), 'utf8'));
const id = manifest.browser_specific_settings.gecko.id;

const base64url = (data) => Buffer.from(data).toString('base64url');

function authorization() {
  const now = Math.floor(Date.now() / 1000);
  const header = base64url(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const payload = base64url(JSON.stringify({ iss: issuer, jti: randomUUID(), iat: now, exp: now + 60 }));
  const signature = createHmac('sha256', secret).update(`${header}.${payload}`).digest('base64url');
  return { Authorization: `JWT ${header}.${payload}.${signature}` };
}

async function get(url) {
  const res = await fetch(url, { headers: authorization() });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${url}`);
  return res;
}

const url = `https://addons.mozilla.org/api/v5/addons/addon/${encodeURIComponent(id)}/versions/v${version}/`;
const { file } = await (await get(url)).json();
console.log(`${version}: ${file?.status ?? 'no file yet'}`);

if (file?.status === 'disabled') process.exit(DISABLED);
if (file?.status !== 'public') process.exit(PENDING);

const buf = Buffer.from(await (await get(file.url)).arrayBuffer());
const hash = `sha256:${createHash('sha256').update(buf).digest('hex')}`;
if (file.hash && file.hash !== hash) throw new Error(`checksum mismatch: expected ${file.hash}, got ${hash}`);
await writeFile(dest, buf);
console.log(`saved    ${dest} (${(buf.length / 1e6).toFixed(1)} MB)`);
