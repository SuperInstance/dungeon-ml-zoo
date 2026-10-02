// conformance.test.mjs — the xruntime-conformance pattern: the vendored core
// must reproduce UPSTREAM byte-exactly. Fixture: results/upstream-hashes.json
// captured from SuperInstance/quilt-dungeons @ 5b99bb3 with its own runner
// CLI (seeds 1-3 × 3 baselines). Plus a live cross-check against the upstream
// clone when it is present next to this repo.
import { test } from 'node:test';
import { readFileSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runGame } from '../core/runner.mjs';
import { loadScript, trajHash } from '../core/runner.mjs';
import { verifyScores } from '../core/scoreboard.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const fixture = JSON.parse(readFileSync(path.join(root, 'results/upstream-hashes.json'), 'utf8'));

for (const r of fixture.runs) {
  test(`conformance vs upstream@5b99bb3: ${r.script} seed ${r.seed}`, async () => {
    const factory = await loadScript(path.join(root, 'core/scripts', `${r.script}.js`));
    const run = runGame(factory, r.seed, { maxTurns: 200 });
    const th = trajHash(run.trajectory);
    if (run.score.score !== r.score) throw new Error(`score drift: ${run.score.score} != ${r.score}`);
    if (th !== r.trajHash) throw new Error(`trajHash drift: ${th} != ${r.trajHash}`);
  });
}

test('live upstream clone byte-identical (when ../quilt-dungeons exists)', { skip: !existsSync(path.resolve(root, '..', 'quilt-dungeons/src/runner.mjs')) }, async () => {
  const upstream = path.resolve(root, '..', 'quilt-dungeons');
  for (const r of [fixture.runs[0], fixture.runs[4], fixture.runs[8]]) {
    const up = execFileSync('node', [path.join(upstream, 'src/runner.mjs'), r.script, String(r.seed)], { encoding: 'utf8' });
    const vend = execFileSync('node', [path.join(root, 'core/runner.mjs'), path.join(root, 'core/scripts', `${r.script}.js`), String(r.seed)], { encoding: 'utf8' });
    if (up !== vend) throw new Error(`live upstream drift on ${r.script} seed ${r.seed}:\nUP: ${up}\nVD: ${vend}`);
  }
});

test('train receipt chain verifies', () => {
  const p = path.join(root, 'results/train-scores.jsonl');
  if (!existsSync(p)) return; // not trained yet
  const v = verifyScores(p);
  if (!v.ok) throw new Error(`train chain broken: ${v.error}`);
});

test('eval receipt chain verifies', () => {
  const p = path.join(root, 'scores.jsonl');
  if (!existsSync(p)) return; // tournament not run yet
  const v = verifyScores(p);
  if (!v.ok) throw new Error(`eval chain broken: ${v.error}`);
});
