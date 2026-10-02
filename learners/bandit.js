// bandit.js — ENTRY 6/9. A contextual bandit that CHOOSES AMONG FIXED SCRIPTS
// (greedy-loot, survivor, hunter, q-table, tiny-nn) per context: phase
// (early/mid/late) × hp-band (4 buckets) = 12 contexts × 5 arms. Pulled on
// TRAIN seeds 1-20 by tools/train-bandit.mjs (each pull = one full episode,
// receipted in results/bandit-pulls.jsonl — pre-registered pulls, honest
// receipt); per-step reward = ΔrunningScore attributed to the current context.
// This file is the frozen selector: argmax mean-reward arm per context,
// delegating each step to that arm's live script instance.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import greedyLoot from '../core/scripts/greedy-loot.js';
import survivor from '../core/scripts/survivor.js';
import hunter from '../core/scripts/hunter.js';
import qTable from './q-table.js';
import tinyNn from './tiny-nn.js';

const here = path.dirname(fileURLToPath(import.meta.url));
let CACHE = null;
export function loadBandit() {
  if (!CACHE) CACHE = JSON.parse(readFileSync(path.join(here, 'bandit-params.json'), 'utf8'));
  return CACHE;
}

export const ARMS = { 'greedy-loot': greedyLoot, 'survivor': survivor, 'hunter': hunter, 'q-table': qTable, 'tiny-nn': tinyNn };

export function contextOf(percept) {
  const t = percept.self.turn;
  const phase = t <= 66 ? 0 : t <= 133 ? 1 : 2;
  const hp = percept.self.hp;
  const hb = hp >= 16 ? 0 : hp >= 11 ? 1 : hp >= 6 ? 2 : 3;
  return phase * 4 + hb; // 0..11
}

export default function bandit(seed) {
  const B = loadBandit();
  const arms = B.arms.map(p => ARMS[p](seed));
  return {
    name: 'bandit',
    step(percept, memory) {
      const c = contextOf(percept);
      const arm = B.choice[c];
      const d = arms[arm].step(percept, memory);
      return { action: d.action, thought: `ctx${c}→${B.arms[arm]}: ${d.thought ?? ''}` };
    },
  };
}
