// tiny-nn.js — ENTRY 5/9. A hand-rolled 2-layer MLP (manual backprop, no
// autodiff, no deps): F features → 12 tanh hidden → 5 action logits, argmax.
// Trained by tools/train-tiny-nn.mjs: behavioral cloning from q-table's EVAL
// seed trajectories (the exposure is disclosed in docs/ZOO.md — the teacher
// never trained on eval seeds, but its eval trajectories are what the student
// clones) + 3 online fine-tune episodes (one-step policy gradient on the sign
// of the shaped reward). Frozen artifact; deterministic argmax at eval.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

export const ACTIONS = ['up', 'down', 'left', 'right', 'wait'];

// 18 percept features (percept only — the same sensory surface as everyone).
export function features(percept) {
  const { x, y, hp, maxHp, turn, loot } = percept.self;
  const r = (percept.local.length - 1) / 2;
  const wall = (dx, dy) => percept.local[dy + r][dx + r] === '#' ? 1 : 0;
  let mD = Infinity, mM = null, iD = Infinity, iI = null;
  for (const m of percept.monsters) {
    const d = Math.max(Math.abs(m.dx), Math.abs(m.dy));
    if (d < mD) { mD = d; mM = m; }
  }
  for (const it of percept.items) {
    const d = Math.max(Math.abs(it.dx), Math.abs(it.dy));
    if (d < iD) { iD = d; iI = it; }
  }
  return [
    hp / maxHp,
    loot / 10,
    turn / 200,
    percept.exitSeen ? 1 : 0,
    percept.exitSeen ? percept.exitSeen.dx / 4 : 0,
    percept.exitSeen ? percept.exitSeen.dy / 4 : 0,
    mM ? mD / 6 : 0, mM ? mM.dx / 6 : 0, mM ? mM.dy / 6 : 0,
    percept.monsters.length / 4,
    iI ? iD / 5 : 0, iI ? iI.dx / 5 : 0, iI ? iI.dy / 5 : 0,
    percept.items.length / 13,
    wall(0, -1), wall(0, 1), wall(-1, 0), wall(1, 0),
  ];
}

let CACHE = null;
export function loadWeights() {
  if (!CACHE) {
    const here = path.dirname(fileURLToPath(import.meta.url));
    CACHE = JSON.parse(readFileSync(path.join(here, 'tinynn-weights.json'), 'utf8'));
  }
  return CACHE;
}

// forward — tanh hidden, linear logits. Exported for the trainer.
export function forward(W, f) {
  const H = W.H, F = W.F;
  const h = new Array(H);
  for (let j = 0; j < H; j++) {
    let s = W.b1[j];
    const row = W.W1[j];
    for (let i = 0; i < F; i++) s += row[i] * f[i];
    h[j] = Math.tanh(s);
  }
  const logits = new Array(5);
  for (let a = 0; a < 5; a++) {
    let s = W.b2[a];
    const row = W.W2[a];
    for (let j = 0; j < H; j++) s += row[j] * h[j];
    logits[a] = s;
  }
  return { h, logits };
}
export function argmax(logits) {
  let a = 0;
  for (let i = 1; i < 5; i++) if (logits[i] > logits[a]) a = i;
  return a;
}

export default function tinyNn(seed) {
  const W = loadWeights();
  return {
    name: 'tiny-nn',
    step(percept) {
      const f = features(percept);
      const { logits } = forward(W, f);
      const a = argmax(logits);
      return { action: ACTIONS[a], thought: `NN→${ACTIONS[a]} z=[${logits.map(v => v.toFixed(1)).join(',')}]` };
    },
  };
}
