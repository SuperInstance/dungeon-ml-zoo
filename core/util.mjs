// VENDORED from SuperInstance/quilt-dungeons @ 5b99bb3 (src/util.mjs) — DO NOT EDIT HERE.
// Provenance + conformance: tests/conformance.test.mjs asserts byte-identical
// trajectories vs upstream on seeds 1-3 (results/upstream-hashes.json fixture).
// quilt-dungeons — receipt utilities (CONTRACT §10).
import { createHash } from 'node:crypto';

// Canonical JSON: keys sorted lexicographically, recursively; arrays keep
// order; no whitespace. Byte-stable for the JSON-serializable values the
// engine and ledger produce.
export function canonicalJSON(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(canonicalJSON).join(',') + ']';
  const keys = Object.keys(value).sort();
  return '{' + keys.map(k => JSON.stringify(k) + ':' + canonicalJSON(value[k])).join(',') + '}';
}

export function sha256Hex(str) {
  return createHash('sha256').update(str, 'utf8').digest('hex');
}
