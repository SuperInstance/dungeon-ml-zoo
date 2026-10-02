// tournament.mjs — THE HONEST INSTRUMENT, executed. Round-robin over the eval
// seeds: every ranked entry plays every eval seed (101-108) through the
// VENDORED runner; every run is receipted in the hash-chained scores.jsonl.
// Fail-closed preconditions: the claims seal must verify AND the battery
// (sha256 of every file the claims bind) must re-derive — otherwise nothing
// runs. A random-policy reference row is receipted but not ranked.
// Post-run: results/eval-metrics.json (the sealed metrics P1-P4 read from
// here) + results/ranking.json (mean ± spread for all 9 entries).
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runGame, trajHash, loadScript } from './core/runner.mjs';
import { appendResults, verifyScores, readScores } from './core/scoreboard.mjs';
import { canonicalJSON, sha256Hex } from './core/util.mjs';
import { EVAL_SEEDS } from './tools/common.mjs';
import { claimsHashOf } from './tools/preregister.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = here;
const P = f => path.join(here, f);

// ---- fail-closed preconditions ---------------------------------------------
const claims = JSON.parse(readFileSync(P('results/claims.json'), 'utf8'));
const seal = JSON.parse(readFileSync(P('results/seal.json'), 'utf8'));
if (claimsHashOf(claims) !== seal.claimsHash) {
  console.error('E_CLAIMS_MODIFIED: claims file does not match the seal — refusing to run the tournament.');
  process.exit(1);
}
const drifted = [];
for (const [f, want] of Object.entries(claims.battery.files)) {
  const got = sha256Hex(readFileSync(P(f), 'utf8'));
  if (got !== want) drifted.push(f);
}
if (drifted.length) {
  console.error('E_BATTERY_MISMATCH: ' + drifted.join(', ') + ' — refusing to run the tournament.');
  process.exit(1);
}

// ---- entries ----------------------------------------------------------------
const ENTRIES = [
  { name: 'greedy-loot', f: 'core/scripts/greedy-loot.js' },
  { name: 'survivor', f: 'core/scripts/survivor.js' },
  { name: 'hunter', f: 'core/scripts/hunter.js' },
  { name: 'q-table', f: 'learners/q-table.js' },
  { name: 'tiny-nn', f: 'learners/tiny-nn.js' },
  { name: 'bandit', f: 'learners/bandit.js' },
  { name: 'ga-params', f: 'learners/ga-params.js' },
  { name: 'ladder', f: 'learners/ladder.js' },
  { name: 'memory-walker', f: 'learners/memory-walker.js' },
];
const REF = [{ name: 'random-policy(ref)', f: 'scripts/random-policy.js' }];

const all = [...ENTRIES, ...REF];
const results = [];
for (const e of all) {
  const factory = await loadScript(P(e.f));
  for (const seed of EVAL_SEEDS) {
    const run = runGame(factory, seed, { maxTurns: 200 });
    results.push({ entry: e, seed, score: run.score.score, turns: run.score.turns, won: run.score.won, trajHash: trajHash(run.trajectory) });
  }
}
appendResults(P('scores.jsonl'), results.map(r => ({
  script: r.entry.name, seed: r.seed, score: r.score, turns: r.turns, won: r.won, trajHash: r.trajHash,
})));
const chain = verifyScores(P('scores.jsonl'));
if (!chain.ok) { console.error('E_CHAIN_BROKEN: ' + chain.error); process.exit(1); }

// ---- ranking ----------------------------------------------------------------
const stats = ENTRIES.map(e => {
  const rs = results.filter(r => r.entry === e);
  const scores = rs.map(r => r.score);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const sd = Math.sqrt(scores.reduce((n, s) => n + (s - mean) ** 2, 0) / scores.length);
  return {
    entry: e.name, mean: +mean.toFixed(2), sd: +sd.toFixed(2),
    spread: +(Math.max(...scores) - Math.min(...scores)).toFixed(2),
    min: Math.min(...scores), max: Math.max(...scores),
    wins: rs.filter(r => r.won).length,
    perSeed: Object.fromEntries(rs.map(r => [r.seed, +r.score.toFixed(1)])),
  };
}).sort((a, b) => b.mean - a.mean);

const get = name => stats.find(s => s.entry === name);
const ref = results.filter(r => r.entry.name === 'random-policy(ref)');
const refBySeed = Object.fromEntries(ref.map(r => [r.seed, r.score]));
const tinynn = results.filter(r => r.entry.name === 'tiny-nn');
const p3 = tinynn.filter(r => r.score > refBySeed[r.seed]).length;
const mw = results.filter(r => r.entry.name === 'memory-walker');

const metrics = {
  'P1.gaMinusQtable': +(get('ga-params').mean - get('q-table').mean).toFixed(4),
  'P2.banditMinusHunter': +(get('bandit').mean - get('hunter').mean).toFixed(4),
  'P3.tinynnBeatsRandomSeeds': p3,
  'P4.memoryWalkerWins': mw.filter(r => r.won).length,
};
const evidence = {
  ranking: stats,
  reference: { name: 'random-policy(ref)', perSeed: Object.fromEntries(Object.entries(refBySeed).map(([k, v]) => [k, +v.toFixed(1)])), mean: +(ref.reduce((n, r) => n + r.score, 0) / ref.length).toFixed(2) },
  memoryWalkerRuns: mw.map(r => ({ seed: r.seed, score: +r.score.toFixed(1), won: r.won, turns: r.turns, trajHash: r.trajHash })),
  tinynnVsRandom: Object.fromEntries(tinynn.map(r => [r.seed, +r.score.toFixed(1)])),
  chain: { records: readScores(P('scores.jsonl')).length, ok: chain.ok },
};

writeFileSync(P('results/eval-metrics.json'), JSON.stringify({ metrics, evidence }, null, 1) + '\n');
writeFileSync(P('results/ranking.json'), JSON.stringify({ ranking: stats, sealedClaims: seal.claimsHash }, null, 1) + '\n');

console.log('FINAL RANKING (eval seeds 101-108, mean ± spread)');
for (const [i, s] of stats.entries()) {
  console.log(`${String(i + 1).padStart(2)}. ${s.entry.padEnd(14)} mean=${String(s.mean).padStart(7)} ±${String(s.spread).padStart(7)} sd=${String(s.sd).padStart(6)} wins(exits)=${s.wins} seeds=${JSON.stringify(s.perSeed)}`);
}
console.log(`ref  random-policy(ref) mean=${evidence.reference.mean} (receipted, not ranked)`);
console.log('P-metrics: ' + canonicalJSON(metrics));
