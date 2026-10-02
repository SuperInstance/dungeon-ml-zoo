// train-q-table.mjs — tabular Q-learning, TRAIN seeds 1-20 ONLY.
// 64 states (threat 0-3 × hp 0-3 × lootV × exitV) × 5 actions; epsilon-greedy
// with a seeded private stream (the world stream is never touched); reward =
// shaped step events + ΔrunningScore. Writes learners/qtable-weights.json
// (the frozen eval artifact) and appends the GREEDY train-seed receipts to
// results/train-scores.jsonl (never scores.jsonl).
import { writeFileSync } from 'node:fs';
import {
  createGame, percepts, step, score, runGame, mulberry32, qState, rewardOf,
  runningScore, ACTIONS, TRAIN_SEEDS, appendTrain, canonicalJSON,
} from './common.mjs';
import qTableScript from '../learners/q-table.js';

const ALPHA = 0.5, GAMMA = 0.9, EPS0 = 0.3, EPS1 = 0.05;
const argmaxRow = row => { let a = 0; for (let i = 1; i < 5; i++) if (row[i] > row[a]) a = i; return a; };

const Q = Array.from({ length: 64 }, () => new Array(5).fill(0));
const episodes = [];
TRAIN_SEEDS.forEach((seed, ep) => {
  const eps = EPS0 + (EPS1 - EPS0) * (ep / (TRAIN_SEEDS.length - 1));
  const rng = mulberry32(0xA11CE + seed * 7919);
  let state = createGame(seed);
  let p = percepts(state);
  let s = qState(p);
  let steps = 0;
  while (state.status === 'active' && state.turn < state.maxTurns) {
    const a = rng() < eps ? Math.floor(rng() * 5) : argmaxRow(Q[s]);
    const before = runningScore(state);
    const res = step(state, ACTIONS[a]);
    state = res.state;
    const r = rewardOf(res.events, runningScore(state) - before);
    p = percepts(state);
    const s2 = qState(p);
    const terminal = state.status !== 'active';
    Q[s][a] += ALPHA * ((r + (terminal ? 0 : GAMMA * Math.max(...Q[s2]))) - Q[s][a]);
    s = s2;
    steps++;
  }
  const out = score(state);
  episodes.push({ seed, eps: +eps.toFixed(3), steps, score: out.score, won: out.won, loot: out.loot, kills: out.kills });
});

writeFileSync(new URL('../learners/qtable-weights.json', import.meta.url),
  JSON.stringify({
    artifact: 'qtable-weights', q: Q, alpha: ALPHA, gamma: GAMMA, eps: [EPS0, EPS1],
    trainedOn: 'seeds 1-20 (train)', states: 64, actions: ACTIONS, episodes,
  }, null, 1));

// GREEDY policy on train seeds = the eval behavior; receipted separately.
for (const seed of TRAIN_SEEDS) appendTrain('q-table(TRAIN)', seed, runGame(qTableScript, seed));

const mean = episodes.reduce((n, e) => n + e.score, 0) / episodes.length;
console.log(JSON.stringify({ tool: 'train-q-table', episodes: episodes.length, meanTrainExploreScore: +mean.toFixed(2), weights: 'learners/qtable-weights.json' }));
