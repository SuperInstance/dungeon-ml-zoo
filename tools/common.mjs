// common.mjs — shared zoo plumbing: re-exports of the VENDORED quilt-dungeons
// core, a seeded RNG, feature extractors, seed lists, and TRAIN-seed receipt
// writing. THE TRAIN/EVAL LAW: train-seed scores go to results/train-scores.jsonl
// ONLY; eval seeds 101-108 go to scores.jsonl ONLY (the official tournament
// chain). Never mixed — that separation IS the generalization honesty.
import { runGame as _runGame, trajHash as _trajHash } from '../core/runner.mjs';
import { createGame, percepts, step, score, DIRS } from '../core/engine.mjs';
import { canonicalJSON, sha256Hex } from '../core/util.mjs';
import { appendResults, readScores, verifyScores, makeRecord, GENESIS } from '../core/scoreboard.mjs';

export { createGame, percepts, step, score, DIRS, canonicalJSON, sha256Hex,
         appendResults, readScores, verifyScores, makeRecord, GENESIS };
export const runGame = _runGame;
export const trajHash = _trajHash;

export const TRAIN_SEEDS = Array.from({ length: 20 }, (_, i) => i + 1);   // 1..20
export const EVAL_SEEDS  = Array.from({ length: 8 },  (_, i) => 101 + i); // 101..108
export const ACTIONS = ['up', 'down', 'left', 'right', 'wait']; // index 0..4

// mulberry32 — the engine's own convention (CONTRACT §2), copied for the
// learners' private exploration streams so the WORLD stream stays untouched.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ (t >>> 14);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// qState — the q-table's sensory compression: threat 0-3 × hp 0-3 × loot
// visible × exit seen = 64 states. threat: no monster in aggro (6) → 0;
// d 4-6 → 1; d 2-3 → 2; d <= 1 → 3. hp: >=16 → 0; >=11 → 1; >=6 → 2; else 3.
// Deliberately inlined (same math) in learners/q-table.js — the fuzz test in
// tests/learners.test.mjs guards the two copies against drift.
// ---------------------------------------------------------------------------
export function threatBucket(percept) {
  let best = Infinity;
  for (const m of percept.monsters) {
    const d = Math.max(Math.abs(m.dx), Math.abs(m.dy));
    if (d < best) best = d;
  }
  if (best === Infinity || best > 6) return 0;
  if (best <= 1) return 3;
  if (best <= 3) return 2;
  return 1;
}
export function hpBucket(hp) {
  if (hp >= 16) return 0;
  if (hp >= 11) return 1;
  if (hp >= 6) return 2;
  return 3;
}
export function qState(percept) {
  const t = threatBucket(percept);
  const h = hpBucket(percept.self.hp);
  const l = percept.items.length > 0 ? 1 : 0;
  const e = percept.exitSeen ? 1 : 0;
  return ((t * 4 + h) * 2 + l) * 2 + e; // 0..63
}
export const Q_STATES = 64;

// runningScore — the score formula (CONTRACT §7) evaluated mid-episode from
// state; used for per-step reward shaping (q-table, tiny-nn fine-tune, bandit).
export function runningScore(state) {
  const kills = state.log.reduce((n, e) => n + (e.type === 'kill' ? 1 : 0), 0);
  return state.loot * 10 + kills * 25 + state.player.hp * 2 - state.turn * 0.1;
}

// Per-step shaped reward from step events (shared by q-table and tiny-nn).
export function rewardOf(events, dr) {
  let r = 0.1 * dr; // loot/kills/hp delta minus turn cost
  for (const e of events) {
    if (e.type === 'pickup') r += e.detail.kind === '$' ? 1.0 : 0.5;
    else if (e.type === 'kill') r += 1.5;
    else if (e.type === 'bump' || e.type === 'invalid-action') r -= 0.1;
    else if (e.type === 'exit') r += 10;
    else if (e.type === 'death') r -= 4;
  }
  return r;
}

// trainPath — where TRAIN receipts live (never scores.jsonl).
export const TRAIN_PATH = new URL('../results/train-scores.jsonl', import.meta.url).pathname;
export const ROOT = new URL('..', import.meta.url).pathname;

export function appendTrain(script, seed, run) {
  return appendResults(TRAIN_PATH, [{
    script, seed, score: run.score.score, turns: run.score.turns,
    won: run.score.won, trajHash: trajHash(run.trajectory),
  }]);
}

// Simple console table for tool output.
export function printTable(rows, header) {
  console.log(header);
  for (const r of rows) console.log(r);
}
