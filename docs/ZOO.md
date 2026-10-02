# ZOO.md — the ML zoo, receipted

Nine entries challenged each other in the vendored `quilt-dungeons @ 5b99bb3`
core. Everything below is measured through the same engine, the same runner,
and the same hash-chained scoreboard; train (seeds 1-20) and eval (seeds
101-108) receipts live in separate chains and were never mixed.

## Final ranking (eval seeds 101-108, official receipts in `scores.jsonl`)

| # | entry | mean | ± spread | sd | exits | per-seed |
|---|-------|------|----------|----|-------|----------|
| 1 | ladder → memory-walker | **144.53** | 99.6 | 33.75 | 0 | 92.4 192 154 174.2 155.5 168.2 102.9 117 |
| 2 | memory-walker | **144.53** | 99.6 | 33.75 | 0 | (byte-identical to ladder: the champion IS the walker) |
| 3 | bandit | 59.54 | 71 | 27.06 | 0 | 39 55 40 101 47.3 41 110 43 |
| 4 | hunter | 57.04 | 79 | 28.43 | 0 | 39 45 40 101 47.3 31 110 43 |
| 5 | greedy-loot | 22.32 | 35.8 | 12.36 | 0 | 20 36 40 20 4.4 24 4.2 30 |
| 6 | survivor | 17.13 | 45.8 | 14.65 | 0 | 20 12 40 20 −3.2 24 −5.8 30 |
| 7 | q-table | 15.88 | 47.3 | 17.33 | 0 | 39 −3.5 20 −2.2 42 17 −5.3 20 |
| 8 | tiny-nn | 2.73 | 25.5 | 10.09 | 0 | −5 −1.2 20 −3 −5.5 −2.7 −0.8 20 |
| 9 | ga-params | 2.24 | 25.5 | 10.34 | 0 | −5 −1.2 20 −3 −5.5 −2.7 −4.7 20 |
| ref | random-policy (receipted, not ranked) | 30.04 | — | — | 0 | |

`verifyScores(scores.jsonl)`: OK, 80 records (9 entries × 8 seeds + 8 ref).

## The entries, mechanism by mechanism

### q-table (tabular Q)
- **Mechanism**: percept → threat bucket 0-3 (no monster/>6 → 0, ≤1 → 3, ≤3 → 2,
  else 1) × hp bucket 0-3 × lootVisible × exitSeen = 64 states × 5 actions.
  ε-greedy (0.3→0.05), α=0.5, γ=0.9, shaped reward (pickup/kill/exit/death +
  0.1·ΔrunningScore). Trained on seeds 1-20 only.
- **Numbers**: explore-train mean 26.81; greedy-train mean 22.05; eval 15.88.
- **Honest limit**: 20 episodes is a toy budget and it shows — the table
  learned that `wait` is safe (583/1070 eval-seed teacher steps were `wait`)
  and never learned to *go anywhere*. Generalization gap: 22.05 → 15.88.

### tiny-nn (2-layer MLP, manual backprop)
- **Mechanism**: 18 percept features → 12 tanh → 5 logits. Behavioral cloning
  from q-table's eval-seed trajectories (1,070 samples; full-batch GD, lr
  0.02, 400 epochs, best-loss snapshot) + 3 online fine-tune episodes
  (seeds 21-23, one-step policy gradient on the sign of shaped reward).
- **Numbers**: BC loss 1.8886 → 0.3753, BC accuracy 91.1%; eval mean 2.73.
- **Honest limit — the imitation trap**: the student faithfully clones a
  teacher that mostly waits, so it inherits the waiting disease and scores
  BELOW the random walker (30.04). It beat random on only 2/8 seeds → P3
  FAIL. Cloning accuracy is not policy quality. (The eval-seed exposure of
  the BC data is disclosed: the teacher never trained on eval seeds, but its
  eval trajectories are what the student saw.)

### bandit (contextual script-selector)
- **Mechanism**: 12 contexts (phase early/mid/late × hp-band 4) × 5 arms
  (greedy-loot, survivor, hunter, q-table, tiny-nn). 100 pre-registered pulls
  (each = one full episode on a train seed, receipted in
  `results/bandit-pulls.jsonl`); per-step reward = ΔrunningScore attributed to
  the live context; argmax mean per context → frozen choice vector.
- **Numbers**: eval 59.54 vs hunter 57.04 → **P2 PASS (+2.5)**. Choice: hunter
  8/12 contexts, survivor 2, q-table 1, tiny-nn 1.
- **Honest limit**: Δscore attribution starves dead arms of signal (a dead
  episode contributes nothing after death); the +2.5 edge is real but thin,
  and came mostly from swapping survivor/q-table into hurt-phase contexts.

### ga-params (GA over a parametric baseline)
- **Mechanism**: a hunter-family script with 4 knobs — aggroRadius,
  fleeThreshold, lootWeight, exploreBias — plus flee, potion-when-hurt, and
  exit-run behaviors. Pop 12, 8 gens, tournament-k3, uniform crossover, seeded
  gaussian mutate, elitism 2, 108 episode-evals. Fitness = mean eval-seed
  score — **the mission-specified leak, disclosed**: the GA saw seeds 101-108
  during evolution.
- **Numbers**: best genome (aggro 3.56, flee 5.55, loot 0.34, explore 0.42),
  fitness 2.24; eval mean 2.24. → **P1 FAIL (−13.64 vs q-table)**.
- **Honest limit**: the receipt is brutal and useful — a GA fitted DIRECTLY on
  the eval seeds still lost to a 20-episode Q-table. The parametric family's
  fitness landscape is nearly flat at ~2.2 (genomes behave near-identically:
  both the hunter-corner prior and the GA's best produce the same four eval
  scores on 101-104), so selection had nothing to climb. Parameter-tuning a
  myopic script does not become exploration.

### ladder (self-play rungs)
- **Mechanism**: rung 0 bar = hunter's train mean (49.55); rungs q-table →
  tiny-nn → ga-params → memory-walker; a rung takes the champion title only by
  strictly beating the holder's train mean.
- **Receipt (which rung beat which)**: q-table 22.05 ✗ · tiny-nn 11.83 ✗ ·
  ga-params 11.93 ✗ · memory-walker **150.07 ✓** (vs 49.55) → champion =
  memory-walker, `learners/ladder.js` delegates to it.
- **Honest limit**: the three learning rungs could not clear the bar their own
  sparring partner set; the ladder crowned the scripted explorer, not a
  learner. That is a real result about the state of the learners, not a
  failure of the ladder.

### memory-walker (the 73-a challenge — NOT a learner)
- **Mechanism**: scripted, RNG-free, percept+memory only. Keeps `obs` (every
  cell ever seen), `walls`, `vis` (stood-on counts), the exit's absolute
  position once seen, and remembered potion locations. Goal: remembered exit >
  nearby potion when hurt (hp ≤ 10, manhattan ≤ 8) > nearest frontier (an
  observed free cell touching never-observed space; BFS distance map, one
  scan). Path: BFS over not-known-wall (unknown = walkable — pathing through
  the unknown IS the discovery). Move filter: avoid cells under a ranged
  monster's clear line or adjacent to melee — unless cornered.
- **Numbers**: train (1-20): 3 exits / 20, mean 150.07. Eval (101-108): mean
  **144.53** — 2.5× hunter — but **0 exits / 8**.
- **Design iterations, receipted honestly** (all measured on train seeds
  only, before sealing):
  - `obs`-frontier (explore never-SEEN, not never-STOOD-ON): the one fix that
    mattered — the vis-based version burned half its turns re-sweeping seen floor.
  - monster-avoiding paths (BFS with visible monsters blocked): REJECTED —
    detours through corridors the walker cannot dodge in; 0 wins, 4 deaths.
  - BSP bottom-right prior (exit room bias): REJECTED — 12 deaths, the
    direction pull trades sweeps for early corridor fights it loses.
  - cross-map potion chase at hp ≤ 12: REJECTED — how walkers die; capped to
    hp ≤ 10 AND manhattan ≤ 8 (kept).
- **Honest limit**: it survives by fighting (bump-attack is the corridor law),
  scores 144 without ever cashing the 200-pt win — and 200 turns is simply
  tight for a wavefront sweep of a 48×24 BSP dungeon through fog radius 4.

## The sealed predictions (preregister@1, seal pushed pre-eval)

| id | claim (sealed verbatim) | measured | verdict |
|----|--------------------------|----------|---------|
| P1 | GA ≥ q-table on eval seeds (leak disclosed) | −13.64 | **FAIL** |
| P2 | bandit-selector ≥ always-hunter | +2.5 | **PASS** |
| P3 | tiny-nn ≥ random-policy on ≥ 7/8 seeds | 2/8 | **FAIL** |
| P4 | memory-walker exits on ≥ 3/8 eval seeds | 0/8 | **FAIL** |

Verdicts scored by the unmodified fleet tool (`tools/preregister.mjs`,
vendored) from `results/eval-metrics.json` against
`sha256:8ec94467…8655bc65`. The tournament refuses to run on claims or
battery drift — it caught one real drift mid-wave (a metrics-nesting fix
after the first seal) and forced an honest re-seal before any official eval
receipt existed.

## The 73-a finding, resolved

The 73-a claim was "exploration memory is the cheapest win on the board."
**Half-true, and the halves are now measured.** Memory doubled-plus the best
baseline's score (144.53 vs 57.04) and took both top ranks — the mean-score
thesis holds. But the exit itself did not fall: 0/8 eval seeds, so P4 FAILs
as sealed, and the 200-pt win bonus remains unclaimed by anything in the
fleet's records (0/24 upstream baseline runs, 0/80 zoo eval runs). What the
zoo adds to 73-a's map: the win is not blocked by memory of walls but by the
turn budget — wavefront sweeping + corridor fighting cannot cover the chain
of BSP rooms in 200 turns. The next lane's opening: corridor-graph memory
(rooms and doorways, not cells) or fight-free pathing, not more parameters.

## Cross-challenge notes for the next lanes

1. The contract is the socket: every entry is a frozen deterministic artifact
   `(seed) => ({name, step(percept, memory)})` — no model sits in the loop;
   the durable artifact is the trajectory + receipt chain.
2. Imitation of a cautious teacher produces a cautious student (tiny-nn);
   reward-shaped tabular learning at n=20 produces a sleeper (q-table). The
   scripted explorer beat them all — and it is also the only entry whose
   design iterations were all measured before sealing.
3. The random walker's 30.04 is a real floor only for agents that move;
   waiting is the dominant local optimum of this reward shape.

Reproduce: `npm run train && node tournament.mjs` (deterministic; the train
chain in `results/train-scores.jsonl` — 123 records — and eval chain in
`scores.jsonl` — 80 records — both verify).
