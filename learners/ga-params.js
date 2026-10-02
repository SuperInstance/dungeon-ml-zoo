// ga-params.js — ENTRY 7/9. A GA over the PARAMETRIC baseline's four knobs:
// aggroRadius, fleeThreshold, lootWeight, exploreBias. Evolved by
// tools/train-ga.mjs (pop 12, 8 gens, tournament-k3, seeded mutate; fitness =
// mean score over eval seeds 101-108 — the mission-specified leak, disclosed
// in docs/ZOO.md). This file is the frozen best genome as an eval artifact.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const ORDER_HF = (dx, dy) => Math.abs(dx) >= Math.abs(dy)
  ? [[Math.sign(dx), 0], [0, Math.sign(dy)], [0, -Math.sign(dy)], [-Math.sign(dx), 0]]
  : [[0, Math.sign(dy)], [Math.sign(dx), 0], [-Math.sign(dx), 0], [0, -Math.sign(dy)]];

export function loadParams() {
  const here = path.dirname(fileURLToPath(import.meta.url));
  return JSON.parse(readFileSync(path.join(here, 'gaparams-weights.json'), 'utf8'));
}

export function parametricScript(P) {
  const memory = { vis: new Map(), exitAbs: null };
  return {
    name: `ga-params(${P.aggroRadius},${P.fleeThreshold},${P.lootWeight.toFixed(2)},${P.exploreBias.toFixed(2)})`,
    step(percept, mem = memory) {
      const { x, y, hp } = percept.self;
      const r = (percept.local.length - 1) / 2;
      const walkable = (dx, dy) => percept.local[dy + r][dx + r] !== '#';
      const vis = k => mem.vis.get(k) ?? 0;
      mem.vis.set(`${x},${y}`, vis(`${x},${y}`) + 1);
      if (percept.exitSeen) mem.exitAbs = { x: x + percept.exitSeen.dx, y: y + percept.exitSeen.dy };

      let mBest = null, iBest = null;
      for (const m of percept.monsters) {
        const d = Math.max(Math.abs(m.dx), Math.abs(m.dy));
        if (!mBest || d < mBest.d) mBest = { ...m, d };
      }
      for (const it of percept.items) {
        const d = Math.abs(it.dx) + Math.abs(it.dy);
        if (!iBest || d < iBest.d) iBest = { ...it, d };
      }

      const stepToward = (tx, ty, why) => {
        const dx = tx - x, dy = ty - y;
        for (const [sx, sy] of ORDER_HF(dx, dy)) {
          if ((sx !== 0 || sy !== 0) && walkable(sx, sy)) return { action: name(sx, sy), thought: why };
        }
        return explore(why + ' (blocked)');
      };
      const name = (dx, dy) => dy < 0 ? 'up' : dy > 0 ? 'down' : dx < 0 ? 'left' : dx > 0 ? 'right' : 'wait';

      const explore = why => {
        if (mem.exitAbs) return stepToward(mem.exitAbs.x, mem.exitAbs.y, why + ' → remembered exit');
        if (P.exploreBias >= 0.5) {
          let best = null;
          for (const [sx, sy] of [[0, -1], [-1, 0], [1, 0], [0, 1]]) {
            if (!walkable(sx, sy)) continue;
            const v = vis(`${x + sx},${y + sy}`);
            if (!best || v < best.v) best = { sx, sy, v };
          }
          if (best) return { action: name(best.sx, best.sy), thought: why + ' (least-visited)' };
        }
        for (const [sx, sy] of [[1, 0], [0, 1], [-1, 0], [0, -1]]) { // right-hand wall-hug
          if (walkable(sx, sy)) return { action: name(sx, sy), thought: why + ' (wall-hug)' };
        }
        return { action: 'wait', thought: why };
      };

      // 1. flee when hurt and threatened
      if (mBest && hp <= P.fleeThreshold) {
        let best = null;
        for (const [sx, sy] of [[0, -1], [-1, 0], [1, 0], [0, 1]]) {
          if (!walkable(sx, sy)) continue;
          let minD = Infinity;
          for (const m of percept.monsters) minD = Math.min(minD, Math.max(Math.abs(x + sx - (x + m.dx)), Math.abs(y + sy - (y + m.dy))));
          if (!best || minD > best.minD) best = { sx, sy, minD };
        }
        if (best) return { action: name(best.sx, best.sy), thought: `flee (${hp}hp ≤ ${P.fleeThreshold})` };
      }
      // 1.5. a visible potion while hurt outranks everything but fleeing
      if (iBest && iBest.kind === '+' && hp <= P.fleeThreshold + 4 && !(mBest && mBest.d <= 1)) {
        return stepToward(x + iBest.dx, y + iBest.dy, `hurt (${hp}hp), potion`);
      }
      // 2. engage: monster inside aggroRadius, unless a high lootWeight wants
      //    the adjacent item first
      if (mBest && mBest.d <= P.aggroRadius && !(iBest && P.lootWeight > 0.5 && iBest.d <= 1)) {
        return { action: name(Math.sign(mBest.dx), Math.sign(mBest.dy)), thought: `engage ${mBest.kind} d=${mBest.d}` };
      }
      // 3. the stairs (or a grab on the way)
      if (mem.exitAbs && !(iBest && P.lootWeight > 0.7 && iBest.d <= 1)) {
        return stepToward(mem.exitAbs.x, mem.exitAbs.y, 'exit run');
      }
      // 4. loot
      if (iBest) return stepToward(x + iBest.dx, y + iBest.dy, `loot ${iBest.kind}`);
      // 5. explore
      return explore('nothing visible');
    },
  };
}

export default function gaParams(seed) {
  return parametricScript(loadParams().best);
}
