// memory-walker.js — ENTRY 9/9 and the 73-a challenge made flesh: NOT a
// learner. A scripted spatial-memory explorer. The 24 upstream baseline runs
// never reached the exit (0/24, the 200-pt win unclaimed) because their world
// ends at the fog radius. This one keeps a map: every observed wall, every
// visited cell, the exit once seen — and walks BFS frontiers into the unknown
// until the stairs are under its feet. Hypothesis (sealed as P4): memory alone
// reaches the exit on >= 3/8 eval seeds. Deterministic by construction: BFS
// with fixed neighbor order, no RNG anywhere.
export const NEIGHBORS = [[0, -1, 'up'], [-1, 0, 'left'], [1, 0, 'right'], [0, 1, 'down']];
const key = (x, y) => x + ',' + y;

export default function memoryWalker(seed) {
  const M = { vis: new Map(), walls: new Set(), obs: new Set(), exitAbs: null, lastM: new Map(), potions: new Map() };

  function observe(percept) {
    const { x, y } = percept.self;
    const r = (percept.local.length - 1) / 2;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const ch = percept.local[dy + r][dx + r];
        M.obs.add(key(x + dx, y + dy)); // seen ≠ stood on: obs is the known map
        if (ch === '#') M.walls.add(key(x + dx, y + dy));
        else if (ch === '>') M.exitAbs = { x: x + dx, y: y + dy };
      }
    }
    if (percept.exitSeen) M.exitAbs = { x: x + percept.exitSeen.dx, y: y + percept.exitSeen.dy };
    for (const m of percept.monsters) M.lastM.set(key(x + m.dx, y + m.dy), percept.self.turn);
    // potions are fuel for the corridor fights: remember where they lie
    const seenPotions = new Set();
    for (const it of percept.items) {
      if (it.kind !== '+') continue;
      const k = key(x + it.dx, y + it.dy);
      M.potions.set(k, { x: x + it.dx, y: y + it.dy });
      seenPotions.add(k);
    }
    const r2 = (percept.local.length - 1) / 2;
    for (let dy = -r2; dy <= r2; dy++) {
      for (let dx = -r2; dx <= r2; dx++) {
        const k = key(x + dx, y + dy);
        if (M.potions.has(k) && !seenPotions.has(k) && percept.local[dy + r2][dx + r2] !== '+') M.potions.delete(k);
      }
    }
  }

  // BFS over "not a known wall" — unknown cells are assumed walkable (and
  // pathing through them is exactly how the map gets discovered). Monsters
  // never block BFS: they move; the danger filter below handles them.
  // Returns the path EXCLUSIVE of start, INCLUSIVE of goal, or null.
  function bfs(sx, sy, isGoal) {
    const start = key(sx, sy);
    const parent = new Map([[start, null]]);
    const q = [[sx, sy]];
    for (let qi = 0; qi < q.length; qi++) {
      const [cx, cy] = q[qi];
      if (qi > 0 && isGoal(cx, cy)) {
        const path = [];
        let k = key(cx, cy);
        while (k && k !== start) { const [px, py] = k.split(',').map(Number); path.unshift([px, py]); k = parent.get(k); }
        return path;
      }
      for (const [dx, dy] of NEIGHBORS) {
        const nx = cx + dx, ny = cy + dy, nk = key(nx, ny);
        if (nx < 0 || ny < 0 || nx > 47 || ny > 23) continue;
        if (M.walls.has(nk) || parent.has(nk)) continue;
        parent.set(nk, key(cx, cy));
        q.push([nx, ny]);
      }
    }
    return null;
  }

  // danger(next): ranged monster aligned within 5 with clear local line, or
  // melee monster adjacent — the two ways this body takes damage.
  function dangerOf(percept, cx, cy) {
    const { x, y } = percept.self;
    const r = (percept.local.length - 1) / 2;
    const at = (ax, ay) => (Math.abs(ax - x) <= r && Math.abs(ay - y) <= r) ? percept.local[ay - y + r][ax - x + r] : null;
    for (const m of percept.monsters) {
      const mx = x + m.dx, my = y + m.dy;
      if (m.kind === 'r' && (mx === cx || my === cy)) {
        const d = Math.max(Math.abs(mx - cx), Math.abs(my - cy));
        if (d >= 1 && d <= 5) {
          let clear = true;
          const sx = Math.sign(cx - mx), sy = Math.sign(cy - my);
          let tx = mx + sx, ty = my + sy;
          while (tx !== cx || ty !== cy) { if (at(tx, ty) === '#') { clear = false; break; } tx += sx; ty += sy; }
          if (clear) return 2;
        }
      }
      if (m.kind === 'm' && Math.max(Math.abs(mx - cx), Math.abs(my - cy)) <= 1) return 2;
    }
    return 0;
  }

  return {
    name: 'memory-walker',
    step(percept) {
      const { x, y, hp, turn } = percept.self;
      observe(percept);
      M.vis.set(key(x, y), (M.vis.get(key(x, y)) ?? 0) + 1);

      // --- goal selection -------------------------------------------------
      let goal = null;
      if (hp <= 10) { // survival detour: NEAREST potion, visible or remembered,
        // but only when it is genuinely close — a cross-map potion chase at
        // half hp through unexplored corridors is how walkers die (measured)
        let best = null;
        for (const { x: px, y: py } of M.potions.values()) {
          const d = Math.abs(px - x) + Math.abs(py - y);
          if (d <= 8 && (!best || d < best.d)) best = { d, gx: px, gy: py };
        }
        for (const it of percept.items) {
          if (it.kind !== '+') continue;
          const d = Math.abs(it.dx) + Math.abs(it.dy);
          if (d <= 8 && (!best || d < best.d)) best = { d, gx: x + it.dx, gy: y + it.dy };
        }
        if (best) goal = { x: best.gx, y: best.gy };
      }
      if (!goal && M.exitAbs && !(M.exitAbs.x === x && M.exitAbs.y === y)) goal = M.exitAbs;
      if (!goal) {
        // frontier: nearest observed cell that touches never-OBSERVED space
        // (obs, not vis — walking over already-seen floor to "explore" is how
        // a walker burns its 200 turns re-sweeping a map it has already seen).
        // No direction prior: measured on train seeds, a bottom-right pull
        // (BSP last-room bias) traded sweeps for early corridor deaths — the
        // neutral wavefront survives AND wins more. Receipted in docs/ZOO.md.
        const dist = new Map([[key(x, y), 0]]);
        const q = [[x, y]];
        for (let qi = 0; qi < q.length; qi++) {
          const [cx, cy] = q[qi];
          for (const [dx, dy] of NEIGHBORS) {
            const nk = key(cx + dx, cy + dy);
            if (cx + dx < 0 || cy + dy < 0 || cx + dx > 47 || cy + dy > 23) continue;
            if (M.walls.has(nk) || dist.has(nk)) continue;
            dist.set(nk, dist.get(key(cx, cy)) + 1);
            q.push([cx + dx, cy + dy]);
          }
        }
        let bestC = null;
        for (const k of M.obs) {
          if (M.walls.has(k)) continue;
          const [cx, cy] = k.split(',').map(Number);
          let touchesUnknown = false;
          for (const [dx, dy] of NEIGHBORS) {
            const nk = key(cx + dx, cy + dy);
            if (cx + dx >= 0 && cy + dy >= 0 && cx + dx <= 47 && cy + dy <= 23 && !M.walls.has(nk) && !M.obs.has(nk)) { touchesUnknown = true; break; }
          }
          if (!touchesUnknown || !dist.has(k)) continue;
          const cost = dist.get(k);
          if (!bestC || cost < bestC.cost) bestC = { cost, gx: cx, gy: cy };
        }
        if (bestC) goal = { x: bestC.gx, y: bestC.gy };
      }

      // --- move selection -------------------------------------------------
      const path = goal ? bfs(x, y, (cx, cy) => cx === goal.x && cy === goal.y) : null;
      const cand = [];
      for (const [dx, dy, name] of NEIGHBORS) {
        const nx = x + dx, ny = y + dy;
        if (M.walls.has(key(nx, ny))) continue; // never bump on purpose
        let dist;
        if (path) {
          const i = path.findIndex(p => p[0] === nx && p[1] === ny);
          dist = i >= 0 ? path.length - 1 - i : path.length; // off-path is worst
        } else {
          dist = goal ? Math.abs(goal.x - nx) + Math.abs(goal.y - ny) : 0;
        }
        cand.push({ name, danger: dangerOf(percept, nx, ny), dist, vis: M.vis.get(key(nx, ny)) ?? 0 });
      }
      if (cand.length === 0) return { action: 'wait', thought: 'boxed in — waiting' };
      cand.sort((a, b) => (a.danger - b.danger) || (goal ? a.dist - b.dist : a.vis - b.vis));
      const pick = cand[0];
      const why = M.exitAbs ? 'walking the remembered map to the stairs'
        : goal ? 'frontier walk into the unknown' : 'no goal — least-visited neighbor';
      return { action: pick.name, thought: `t${turn} hp${hp} ${why} (d=${pick.danger}, vis=${pick.vis})` };
    },
  };
}
