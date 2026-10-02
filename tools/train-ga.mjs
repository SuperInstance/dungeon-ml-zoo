// train-ga.mjs — GA over the parametric baseline's four knobs (aggroRadius,
// fleeThreshold, lootWeight, exploreBias). Pop 12, 8 gens, tournament-k3
// select, uniform crossover, seeded gaussian mutate, elitism 2. FITNESS =
// mean score over EVAL seeds 101-108 — the mission-specified leak (the GA
// sees the eval seeds during evolution; its eval numbers are optimistic by
// construction and that is receipted in docs/ZOO.md and in P1's refusal
// branch). Deterministic: one seeded stream drives everything.
import { writeFileSync } from 'node:fs';
import {
  createGame, percepts, step, score, runGame, mulberry32,
  TRAIN_SEEDS, EVAL_SEEDS, appendTrain, canonicalJSON, trajHash,
} from './common.mjs';
import gaParams, { parametricScript } from '../learners/ga-params.js';

const POP = 12, GENS = 8, K = 3, ELITE = 2;
const rng = mulberry32(424242);
const gauss = () => {
  const u = Math.max(rng(), 1e-12), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

function runParametric(g, seed) {
  const script = parametricScript(g);
  const memory = {};
  let state = createGame(seed);
  const trajectory = [];
  while (state.status === 'active' && state.turn < state.maxTurns) {
    const p = percepts(state);
    let action = null, thought = '';
    try {
      const d = script.step(p, memory);
      if (d && d.action != null) action = d.action;
      if (d && d.thought != null) thought = String(d.thought);
    } catch { action = null; thought = 'script-error'; }
    const res = step(state, action);
    state = res.state;
    trajectory.push({ turn: state.turn, action, thought, hp: state.player.hp, events: res.events });
  }
  return { trajectory, out: score(state) };
}

const fitness = g => EVAL_SEEDS.reduce((n, s) => n + runParametric(g, s).out.score, 0) / EVAL_SEEDS.length;

const clamp = (g) => ({
  aggroRadius: Math.min(6, Math.max(2, g.aggroRadius)),
  fleeThreshold: Math.min(14, Math.max(0, g.fleeThreshold)),
  lootWeight: Math.min(1, Math.max(0, g.lootWeight)),
  exploreBias: Math.min(1, Math.max(0, g.exploreBias)),
});
const mutate = (g) => clamp({
  aggroRadius: g.aggroRadius + gauss() * 0.5,
  fleeThreshold: g.fleeThreshold + gauss() * 1.5,
  lootWeight: g.lootWeight + gauss() * 0.12,
  exploreBias: g.exploreBias + gauss() * 0.12,
});
const crossover = (a, b) => clamp(Object.fromEntries(
  ['aggroRadius', 'fleeThreshold', 'lootWeight', 'exploreBias'].map(k => [k, rng() < 0.5 ? a[k] : b[k]])));
const tournament = (pop, fit) => {
  let best = null;
  for (let i = 0; i < K; i++) {
    const c = pop[Math.floor(rng() * pop.length)];
    if (!best || fit(c) > fit(best)) best = c;
  }
  return best;
};

// Initial population seeded AROUND the strongest baseline (hunter's corner:
// high aggro, low flee, mid loot) — a documented prior, not a hidden answer:
// every genome is still perturbed by the seeded stream and the GA can (and
// does) walk away from the corner if fitness says so.
let pop = Array.from({ length: POP }, () => clamp({
  aggroRadius: 3.5 + 2.5 * rng(), fleeThreshold: rng() * 10,
  lootWeight: 0.2 + 0.6 * rng(), exploreBias: rng(),
}));
const history = [];
let evals = 0;
for (let gen = 0; gen < GENS; gen++) {
  const fits = pop.map(g => { evals++; return fitness(g); });
  const ranked = pop.map((g, i) => ({ g, f: fits[i] })).sort((a, b) => b.f - a.f);
  history.push({ gen, best: +ranked[0].f.toFixed(2), mean: +(fits.reduce((a, b) => a + b, 0) / POP).toFixed(2), bestGenome: ranked[0].g });
  const next = ranked.slice(0, ELITE).map(r => r.g); // elitism
  while (next.length < POP) {
    const p1 = tournament(pop, g => fits[pop.indexOf(g)]);
    const p2 = tournament(pop, g => fits[pop.indexOf(g)]);
    next.push(mutate(crossover(p1, p2)));
  }
  pop = next;
}
const finalFits = pop.map(g => fitness(g)); evals += POP;
const bestIdx = finalFits.indexOf(Math.max(...finalFits));
const best = pop[bestIdx];

writeFileSync(new URL('../learners/gaparams-weights.json', import.meta.url),
  JSON.stringify({
    artifact: 'gaparams-weights', best, fitness: +finalFits[bestIdx].toFixed(2),
    ga: { pop: POP, gens: GENS, tournamentK: K, elite: ELITE, fitness: 'mean eval-seed score (leak disclosed)', episodeEvals: evals },
    history,
  }, null, 1));

for (const seed of TRAIN_SEEDS) {
  const { trajectory, out } = runParametric(best, seed);
  appendResultsTrain(seed, out, trajectory);
}
function appendResultsTrain(seed, out, trajectory) {
  appendTrain('ga-params(TRAIN)', seed, { trajectory, score: out });
}
console.log(JSON.stringify({ tool: 'train-ga', best, fitness: +finalFits[bestIdx].toFixed(2), episodeEvals: evals }));
