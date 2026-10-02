// train-bandit.mjs — contextual bandit over FIXED SCRIPT ARMS. Contexts:
// phase(early<=66, mid<=133, late) × hp-band(4) = 12; arms: greedy-loot,
// survivor, hunter, q-table, tiny-nn. THE PULLS ARE PRE-REGISTERED: every
// pull is one full episode of one arm on one TRAIN seed (1-20), receipted in
// results/bandit-pulls.jsonl {seed, arm, score, won, trajHash}. Reward per
// step = ΔrunningScore attributed to the current context; arm* = argmax mean
// reward per context. Writes learners/bandit-params.json + selector receipts.
import { writeFileSync, appendFileSync } from 'node:fs';
import {
  createGame, percepts, step, score, runGame, runningScore, trajHash,
  TRAIN_SEEDS, appendTrain, canonicalJSON,
} from './common.mjs';
import greedyLoot from '../core/scripts/greedy-loot.js';
import survivor from '../core/scripts/survivor.js';
import hunter from '../core/scripts/hunter.js';
import qTable from '../learners/q-table.js';
import tinyNn from '../learners/tiny-nn.js';
import banditScript, { contextOf } from '../learners/bandit.js';

const ARMS = ['greedy-loot', 'survivor', 'hunter', 'q-table', 'tiny-nn'];
const FACTORY = { 'greedy-loot': greedyLoot, 'survivor': survivor, 'hunter': hunter, 'q-table': qTable, 'tiny-nn': tinyNn };

const sum = Array.from({ length: 12 }, () => new Array(5).fill(0));
const cnt = Array.from({ length: 12 }, () => new Array(5).fill(0));
const pulls = [];

for (const seed of TRAIN_SEEDS) {
  for (let ai = 0; ai < ARMS.length; ai++) {
    const arm = ARMS[ai];
    const script = FACTORY[arm](seed);
    const memory = {};
    let state = createGame(seed);
    const trajectory = [];
    while (state.status === 'active' && state.turn < state.maxTurns) {
      const p = percepts(state);
      const c = contextOf(p);
      let action = null, thought = '';
      try {
        const d = script.step(p, memory);
        if (d && d.action != null) action = d.action;
        if (d && d.thought != null) thought = String(d.thought);
      } catch { action = null; thought = 'script-error'; }
      const before = runningScore(state);
      const res = step(state, action);
      state = res.state;
      sum[c][ai] += runningScore(state) - before;
      cnt[c][ai] += 1;
      trajectory.push({ turn: state.turn, action, thought, hp: state.player.hp, events: res.events });
    }
    const out = score(state);
    pulls.push({ seed, arm, score: out.score, won: out.won, trajHash: trajHash(trajectory) });
  }
}
for (const p of pulls) appendFileSync(new URL('../results/bandit-pulls.jsonl', import.meta.url), canonicalJSON(p) + '\n');

const means = sum.map((row, c) => row.map((s, ai) => (cnt[c][ai] ? s / cnt[c][ai] : -Infinity)));
const choice = means.map(row => row.reduce((b, v, i) => (v > row[b] ? i : b), 0));

writeFileSync(new URL('../learners/bandit-params.json', import.meta.url),
  JSON.stringify({
    artifact: 'bandit-params', arms: ARMS, choice,
    means: means.map(r => r.map(v => +v.toFixed(4))), pullCounts: cnt, trainedOn: 'seeds 1-20 (train), 100 pulls',
  }, null, 1));

for (const seed of TRAIN_SEEDS) appendTrain('bandit(TRAIN)', seed, runGame(banditScript, seed));
const fromCounts = Object.fromEntries(ARMS.map((a, i) => [a, choice.filter(c => c === i).length]));
console.log(JSON.stringify({ tool: 'train-bandit', pulls: pulls.length, armWins: fromCounts, choice }));
