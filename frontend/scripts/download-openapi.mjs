import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

const url = process.env.VESTIGIUM_OPENAPI_URL ?? 'http://localhost:8008/v3/api-docs';
const outFile = 'openapi/vestigium-openapi.json';

async function main() {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch OpenAPI from ${url}: HTTP ${res.status}`);
  }
  const json = await res.json();
  await mkdir(dirname(outFile), { recursive: true });
  await writeFile(outFile, JSON.stringify(json, null, 2) + '\n', 'utf-8');
  // eslint-disable-next-line no-console
  console.log(`Saved OpenAPI spec to ${outFile}`);
}

await main();


