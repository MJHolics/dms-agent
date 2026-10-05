// 시연 재생이 자막대로 판단되는지: 깜빡임은 경보 없음, 하품·옆 보기는 주의, 겹치면 위험, 2초 감으면 경보.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Monitor } from '../decision.js';
import { SCENARIO_SEC, observe } from '../scenario.js';

function run(fps) {
  const mon = new Monitor();
  const out = [];
  for (let t = 0; t < SCENARIO_SEC; t += 1 / fps) out.push({ t, ...mon.update(t, observe(t)) });
  return out;
}
const maxLevel = (rows, a, b) => Math.max(...rows.filter((r) => r.t >= a && r.t < b).map((r) => r.alert_level));

for (const fps of [60, 30, 15, 8]) {
  test(`시연 재생 ${fps}fps`, () => {
    const rows = run(fps);
    assert.equal(maxLevel(rows, 2.3, 7.4), 0, '깜빡임 구간에 경보가 없어야 한다');
    assert.equal(maxLevel(rows, 8.2, 9.4), 1, '하품은 주의');
    assert.equal(maxLevel(rows, 12.2, 13.4), 1, '옆 보기는 주의');
    assert.equal(maxLevel(rows, 16.3, 17.8), 2, '하품+옆 보기는 위험');
    assert.equal(maxLevel(rows, 18.6, 22.9), 0, '감은 지 2초 전에는 경보가 없어야 한다');
    const first = rows.find((r) => r.alert_level === 3);
    assert.ok(first && first.t >= 23.0 && first.t <= 23.0 + 0.1 + 2 / fps, `경보 시각 ${first && first.t}`);
    assert.equal(maxLevel(rows, 25.3, 30), 0, '눈을 뜨면 풀린다');
  });
}
