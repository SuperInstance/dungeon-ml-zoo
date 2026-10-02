// learners.test.mjs — the zoo's own invariants: determinism of every entry,
// no drift between the two qState copies, weights sanity, chain integrity.
import { test } from 'node:test';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runGame, trajHash, loadScript } from '../core/runner.mjs';
import { verifyScores } from '../core/scoreboard.mjs';
import { qState as qStateCommon } from '../tools/common.mjs';
import { qState as qStateLearner } from '../learners/q-table.js';
import { forward, argmax, features } from '../learners/tiny-nn.js';
import { contextOf, loadBandit, ARMS } from '../learners/bandit.js';
import { loadParams } from '../learners/ga-params.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

test('qState copies agree (no drift between tool + learner)', () => {
  const mk = (md, hp, items, exit) => ({
    self: { hp, maxHp: 20, x: 24, y: 12, turn: 0, loot: 0 },
    local: ['.........', '.........', '.........', '.........', '....@....', '.........', '.........', '.........', '.........'],
    monsters: md, items, exitSeen: exit,
  });
  for (const d of [0, 1, 2, 3, 4, 5, 6, 7, 9]) {
    for (const hp of [1, 6, 11, 16, 20]) {
      for (const items of [[], [{ dx: 1, dy: 1, kind: '$' }]]) {
        for (const exit of [null, { dx: 2, dy: 2 }]) {
          const p = mk(d === 0 ? [] : [{ dx: d, dy: 0, kind: 'm', hp: 5 }], hp, items, exit);
          if (qStateCommon(p) !== qStateLearner(p)) throw new Error(`drift at d=${d} hp=${hp}`);
          if (qStateLearner(p) < 0 || qStateLearner(p) > 63) throw new Error(`state out of range at d=${d} hp=${hp}`);
        }
      }
    }
  }
});

const ENTRIES = [
  'core/scripts/greedy-loot.js', 'core/scripts/survivor.js', 'core/scripts/hunter.js',
  'learners/q-table.js', 'learners/tiny-nn.js', 'learners/bandit.js',
  'learners/ga-params.js', 'learners/ladder.js', 'learners/memory-walker.js',
];

for (const e of ENTRIES) {
  test(`determinism: ${e} replays byte-identical`, { skip: !existsSync(path.join(root, e)) }, async () => {
    const factory = await loadScript(path.join(root, e));
    const a = runGame(factory, 1);
    const b = runGame(factory, 1);
    if (trajHash(a.trajectory) !== trajHash(b.trajectory)) throw new Error('non-deterministic replay');
    if (a.score.score !== b.score.score) throw new Error('score drift across replays');
  });
}

test('tiny-nn forward: 5 finite logits on a synthetic percept', () => {
  if (!existsSync(path.join(root, 'learners/tinynn-weights.json'))) return;
  const p = {
    self: { hp: 13, maxHp: 20, x: 5, y: 5, turn: 40, loot: 2 },
    local: ['#########', '#...$...#', '#.m.....#', '#.......#', '#...@..>#', '#.......#', '#.......#', '#...+...#', '#########'],
    monsters: [{ dx: -2, dy: -1, kind: 'm', hp: 5 }],
    items: [{ dx: -1, dy: 1, kind: '$' }, { dx: 1, dy: 3, kind: '+' }],
    exitSeen: { dx: 2, dy: 0 },
  };
  const f = features(p);
  if (f.length !== 18) throw new Error(`features length ${f.length}`);
  const { logits } = forward(JSON.parse(readFileSync(path.join(root, 'learners/tinynn-weights.json'), 'utf8')), f);
  if (logits.length !== 5 || logits.some(v => !Number.isFinite(v))) throw new Error('bad logits');
  if (argmax(logits) < 0 || argmax(logits) > 4) throw new Error('bad argmax');
});

test('bandit params: 12 contexts, valid arms', () => {
  if (!existsSync(path.join(root, 'learners/bandit-params.json'))) return;
  const B = loadBandit();
  if (B.choice.length !== 12) throw new Error(`choice length ${B.choice.length}`);
  if (B.arms.length !== 5 || B.arms.some(a => !ARMS[a])) throw new Error('bad arms');
  if (B.choice.some(c => c < 0 || c > 4)) throw new Error('arm index out of range');
  if (contextOf({ self: { turn: 0, hp: 20 } }) !== 0 || contextOf({ self: { turn: 199, hp: 1 } }) !== 11) throw new Error('context mapping');
});

test('ga best genome within ranges', () => {
  if (!existsSync(path.join(root, 'learners/gaparams-weights.json'))) return;
  const P = loadParams().best;
  if (P.aggroRadius < 2 || P.aggroRadius > 6) throw new Error('aggroRadius');
  if (P.fleeThreshold < 0 || P.fleeThreshold > 14) throw new Error('fleeThreshold');
  if (P.lootWeight < 0 || P.lootWeight > 1) throw new Error('lootWeight');
  if (P.exploreBias < 0 || P.exploreBias > 1) throw new Error('exploreBias');
});

test('memory-walker is RNG-free by construction (module source scan)', () => {
  const src = readFileSync(path.join(root, 'learners/memory-walker.js'), 'utf8');
  if (/Math\.random|mulberry|Date\.now|process\.hrtime/.test(src)) throw new Error('memory-walker contains a nondeterminism source');
});
