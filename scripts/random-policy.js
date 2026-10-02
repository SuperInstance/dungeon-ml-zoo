// random-policy.js — REFERENCE FLOOR, not a ranked entry. Uniform random walk
// over the 5 actions with a per-seed private stream (deterministic per seed).
// Exists because pre-registered P3 needed a floor to clear: tiny-nn vs random.
export const ACTIONS = ['up', 'down', 'left', 'right', 'wait'];
export default function randomPolicy(seed) {
  let a = (seed ^ 0x5EEDC0DE) >>> 0;
  const next = () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ (t >>> 14);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    name: 'random-policy(ref)',
    step() {
      return { action: ACTIONS[Math.floor(next() * 5)], thought: 'random walk' };
    },
  };
}
