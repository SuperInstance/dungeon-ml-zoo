# dungeon-ml-zoo

The ML zoo in SuperInstance/quilt-dungeons: **five learners + a memory-walker
challenging each other** on the same deterministic dungeon the fleet's
baselines could not solve (0/24 runs ever reached the 200-pt exit).

Core contract VENDORED from `SuperInstance/quilt-dungeons @ 5b99bb3` into
`core/` (provenance header on every file); `tests/conformance.test.mjs`
asserts byte-identical trajectories vs upstream on seeds 1-3.

## The nine entries

| # | entry | kind | file |
|---|-------|------|------|
| 1-3 | greedy-loot, survivor, hunter | upstream baselines (sparring partners) | `core/scripts/` |
| 4 | q-table | tabular Q, 64 states × 5 actions, ε-greedy train seeds 1-20 | `learners/q-table.js` |
| 5 | tiny-nn | hand-rolled 2-layer MLP (manual backprop), BC + 3 fine-tune eps | `learners/tiny-nn.js` |
| 6 | bandit | contextual bandit choosing among 5 fixed scripts, 12 contexts | `learners/bandit.js` |
| 7 | ga-params | GA (pop 12 × 8 gens) over 4 parametric-baseline knobs | `learners/ga-params.js` |
| 8 | ladder | self-play rungs: each champion is the next's sparring partner | `learners/ladder.js` |
| 9 | memory-walker | NOT a learner: scripted spatial-memory frontier explorer | `learners/memory-walker.js` |

(ref: `scripts/random-policy.js` — receipted floor for P3, not ranked.)

## The honesty instrument

Predictions P1-P4 written, **sealed and pushed BEFORE the final eval run**
(the ritual), scored afterward by `tools/preregister.mjs` (vendored from
fleet-seeds, unmodified). `tournament.mjs` refuses to run if the claims or
the artifact battery (sha256 of every bound file) drift from the seal.

- TRAIN seeds **1-20** → `results/train-scores.jsonl` (its own hash chain)
- EVAL seeds **101-108** → `scores.jsonl` (the official chain)
- the two are never mixed

## Run

```sh
node --test tests/          # conformance + zoo invariants
npm run train               # q-table → tiny-nn → bandit → ga → ladder
node tournament.mjs         # eval round-robin + sealed metrics (post-seal)
```

Zero model calls, zero network, Node >= 20 stdlib only. Numbers, mechanisms
and honest limits per learner: `docs/ZOO.md`.
