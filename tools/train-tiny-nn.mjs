// train-tiny-nn.mjs — behavioral cloning + online fine-tune for the 2-layer
// MLP (18→12 tanh→5). BC data: q-table's EVAL-seed trajectories (seeds
// 101-108) — the exposure this gives the student is disclosed in docs/ZOO.md.
// Fine-tune: 3 episodes (seeds 21-23) with one-step policy-gradient updates
// on the sign of the shaped reward. Deterministic: seeded init, seeded batch
// shuffle. Writes learners/tinynn-weights.json + greedy train receipts.
import { writeFileSync } from 'node:fs';
import {
  createGame, percepts, step, score, runGame, mulberry32, rewardOf,
  runningScore, ACTIONS, TRAIN_SEEDS, EVAL_SEEDS, appendTrain, trajHash,
} from './common.mjs';
import qTableScript from '../learners/q-table.js';
import tinyNnScript, { features, forward, argmax } from '../learners/tiny-nn.js';

const F = 18, H = 12, EPOCHS = 400, BATCH = 4096, LR = 0.02, FT_LR = 0.005, FT_SEEDS = [21, 22, 23]; // full-batch GD: stable on 1070 samples
const rng = mulberry32(0xF00D5EED);
const gauss = () => { // Box-Muller with the seeded stream
  const u = Math.max(rng(), 1e-12), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const W = {
  F, H,
  W1: Array.from({ length: H }, () => Array.from({ length: F }, () => gauss() * 0.5)),
  b1: new Array(H).fill(0),
  W2: Array.from({ length: 5 }, () => Array.from({ length: H }, () => gauss() * 0.5)),
  b2: new Array(5).fill(0),
};

// ---- collect BC data from the frozen q-table on EVAL seeds -----------------
const data = [];
const bcCheck = [];
for (const seed of EVAL_SEEDS) {
  let state = createGame(seed);
  let p = percepts(state);
  const script = qTableScript(seed);
  while (state.status === 'active' && state.turn < state.maxTurns) {
    const d = script.step(p, {});
    data.push({ f: features(p), a: ACTIONS.indexOf(d.action) });
    const res = step(state, d.action);
    state = res.state;
    p = percepts(state);
  }
  const out = score(state);
  const direct = runGame(qTableScript, seed);
  if (Math.abs(out.score - direct.score.score) > 1e-9) throw new Error(`BC replay mismatch seed ${seed}`);
  bcCheck.push({ seed, score: out.score, trajHash: trajHash(direct.trajectory) });
}

// ---- manual backprop --------------------------------------------------------
const softmax = z => {
  const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
};
function backward(batch, lr, tag) { // tag: {ce:true} for cross-entropy on s.a, {coef} for policy gradient
  const gW1 = Array.from({ length: H }, () => new Array(F).fill(0));
  const gb1 = new Array(H).fill(0);
  const gW2 = Array.from({ length: 5 }, () => new Array(H).fill(0));
  const gb2 = new Array(5).fill(0);
  let loss = 0, correct = 0;
  for (const s of batch) {
    const { h, logits } = forward(W, s.f);
    const pr = softmax(logits);
    if (pr[s.a] > 1e-12) loss += -Math.log(pr[s.a]);
    if (argmax(logits) === s.a) correct++;
    const dz = pr.map((v, i) => (tag.ce ? v - (i === s.a ? 1 : 0) : tag.coef * ((i === s.a ? 1 : 0) - v)))
      .map(v => Math.max(-2, Math.min(2, v))); // gradient clip: BC from a 20-episode teacher must not blow up
    for (let a = 0; a < 5; a++) {
      gb2[a] += dz[a];
      for (let j = 0; j < H; j++) gW2[a][j] += dz[a] * h[j];
    }
    const dh = new Array(H).fill(0);
    for (let j = 0; j < H; j++) {
      let s2 = 0;
      for (let a = 0; a < 5; a++) s2 += dz[a] * W.W2[a][j];
      dh[j] = s2 * (1 - h[j] * h[j]); // tanh'
    }
    for (let j = 0; j < H; j++) {
      gb1[j] += dh[j];
      for (let i = 0; i < F; i++) gW1[j][i] += dh[j] * s.f[i];
    }
  }
  const n = batch.length;
  for (let j = 0; j < H; j++) {
    W.b1[j] -= lr * gb1[j] / n;
    for (let i = 0; i < F; i++) W.W1[j][i] -= lr * gW1[j][i] / n;
  }
  for (let a = 0; a < 5; a++) {
    W.b2[a] -= lr * gb2[a] / n;
    for (let j = 0; j < H; j++) W.W2[a][j] -= lr * gW2[a][j] / n;
  }
  return { loss: loss / n, acc: correct / n };
}

const loss0 = backward(data, 0, { coef: 0 }).loss; // measure before training
let best = { loss: loss0, acc: 0, W: JSON.parse(JSON.stringify(W)) };
let last = { loss: loss0, acc: 0 };
for (let ep = 0; ep < EPOCHS; ep++) {
  const order = data.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  for (let b = 0; b < order.length; b += BATCH) last = backward(order.slice(b, b + BATCH).map(i => data[i]), LR, { ce: true });
  if (last.loss < best.loss) best = { loss: last.loss, acc: last.acc, W: JSON.parse(JSON.stringify(W)) };
}
Object.assign(W, best.W); // snapshot weights with the lowest BC loss survive

// ---- 3 online fine-tune episodes (one-step policy gradient) -----------------
const ft = [];
for (const seed of FT_SEEDS) {
  let state = createGame(seed);
  let p = percepts(state);
  const trace = [];
  const trajectory = [];
  while (state.status === 'active' && state.turn < state.maxTurns) {
    const f = features(p);
    const a = argmax(forward(W, f).logits);
    trace.push({ f, a });
    const before = runningScore(state);
    const res = step(state, ACTIONS[a]);
    state = res.state;
    const r = rewardOf(res.events, runningScore(state) - before);
    trace[trace.length - 1].r = r;
    trajectory.push({ turn: state.turn, action: ACTIONS[a], thought: 'fine-tune step', hp: state.player.hp, events: res.events });
    p = percepts(state);
  }
  for (const { f, a, r } of trace) backward([{ f, a }], FT_LR, { coef: Math.max(-1, Math.min(1, r)) });
  const out = score(state);
  ft.push({ seed, score: out.score, won: out.won, steps: trace.length });
  appendTrain('tiny-nn(TRAIN-FT)', seed, { trajectory, score: out });
}

writeFileSync(new URL('../learners/tinynn-weights.json', import.meta.url),
  JSON.stringify({
    artifact: 'tinynn-weights', ...W,
    trainedOn: { bc: 'q-table eval trajectories (seeds 101-108)', fineTune: FT_SEEDS, epochs: EPOCHS, lr: LR, ftLr: FT_LR },
    receipts: { loss0: +loss0.toFixed(4), bestLoss: +best.loss.toFixed(4), bestBcAccuracy: +best.acc.toFixed(4), ft, bcCheck },
  }, null, 1));

for (const seed of TRAIN_SEEDS) appendTrain('tiny-nn(TRAIN)', seed, runGame(tinyNnScript, seed));
console.log(JSON.stringify({ tool: 'train-tiny-nn', bcSamples: data.length, loss0: +loss0.toFixed(4), bestLoss: +best.loss.toFixed(4), bestBcAccuracy: +best.acc.toFixed(4), finalLoss: +last.loss.toFixed(4) }));
