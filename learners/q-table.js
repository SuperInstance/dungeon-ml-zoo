// q-table.js — ENTRY 4/9. Tabular Q-learning over the discretized percept:
// threat 0-3 × hp 0-3 × lootVisible × exitSeen = 64 states × 5 actions.
// Trained by tools/train-q-table.mjs on TRAIN seeds 1-20 ONLY (epsilon-greedy,
// seeded exploration); this file is the FROZEN eval artifact: pure greedy
// lookup, fully deterministic, generalization tested on eval seeds 101-108.
// Train-seed scores are receipted in results/train-scores.jsonl, never mixed.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

export const ACTIONS = ['up', 'down', 'left', 'right', 'wait'];

export function qState(percept) {
  let best = Infinity;
  for (const m of percept.monsters) {
    const d = Math.max(Math.abs(m.dx), Math.abs(m.dy));
    if (d < best) best = d;
  }
  const t = (best === Infinity || best > 6) ? 0 : best <= 1 ? 3 : best <= 3 ? 2 : 1;
  const hp = percept.self.hp;
  const h = hp >= 16 ? 0 : hp >= 11 ? 1 : hp >= 6 ? 2 : 3;
  const l = percept.items.length > 0 ? 1 : 0;
  const e = percept.exitSeen ? 1 : 0;
  return ((t * 4 + h) * 2 + l) * 2 + e; // 0..63
}

// Weights are read lazily at factory time so training tools can import this
// module before learners/qtable-weights.json exists.
let CACHE = null;
export function loadWeights() {
  if (!CACHE) {
    const here = path.dirname(fileURLToPath(import.meta.url));
    CACHE = JSON.parse(readFileSync(path.join(here, 'qtable-weights.json'), 'utf8'));
  }
  return CACHE;
}

export default function qTableScript(seed) {
  const W = loadWeights();
  if (!W || !Array.isArray(W.q) || W.q.length !== 64) throw new Error('qtable-weights.json missing/corrupt');
  return {
    name: 'q-table',
    step(percept) {
      const s = qState(percept);
      const row = W.q[s];
      let a = 0;
      for (let i = 1; i < 5; i++) if (row[i] > row[a]) a = i; // greedy; ties → lowest index
      return { action: ACTIONS[a], thought: `Q[${s}]→${ACTIONS[a]} v=${row[a].toFixed(2)}` };
    },
  };
}
