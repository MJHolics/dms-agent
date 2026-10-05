// 파이썬 원본(agents/decision.py)이 낸 답과 JS 이식본이 낸 답을 같은 입력으로 대조한다.
// fixture.json은 `python tools/export_parity_fixture.py`가 만든다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as D from '../decision.js';

const fx = JSON.parse(readFileSync(new URL('./fixture.json', import.meta.url), 'utf8'));
const TOL = 1e-9;
const close = (a, b, what) => assert.ok(Math.abs(a - b) <= TOL, `${what}: js ${a} vs py ${b}`);

test('EAR·MAR 기하 계산', () => {
  for (const [i, g] of fx.geometry.entries()) {
    close(D.ear(g.eye), g.ear, `ear #${i}`);
    close(D.mar(g.mouth), g.mar, `mar #${i}`);
  }
});

test('변환 행렬 → 고개 각도', () => {
  for (const [i, a] of fx.angles.entries()) {
    const [pitch, yaw] = D.headAngles(a.matrix);
    close(pitch, a.pitch, `pitch #${i}`);
    close(yaw, a.yaw, `yaw #${i}`);
  }
});

test('눈 감김 점수 + 눈 주변 근육 → 뜬 정도', () => {
  for (const [i, o] of fx.openness.entries()) {
    const [bl, br, es, cs, ms] = o.args;
    const sq = D.squintScore(es, cs, ms);
    close(sq, o.squint, `squint #${i}`);
    close(D.eyeOpenness(bl, br, sq), o.open, `open #${i}`);
  }
});

test('신호 → 위험 플래그 → 경보 레벨', () => {
  for (const [i, d] of fx.decisions.entries()) {
    const flags = D.classify(d.input);
    assert.deepEqual(flags, d.flags, `flags #${i}`);
    const [level, reason] = D.alert(flags, d.input.detected_objects);
    assert.equal(level, d.level, `level #${i}`);
    assert.equal(reason, d.reason, `reason #${i}`);
  }
});

for (const seq of fx.sequences) {
  test(`시퀀스 전체 — ${seq.name}`, () => {
    const mon = seq.mode === 'blink'
      ? new D.Monitor(true, D.BLINK_OPEN_RATIO, D.BLINK_OPEN_RANGE, D.BLINK_OPEN_THRESH)
      : new D.Monitor();
    for (const [i, f] of seq.frames.entries()) {
      const got = mon.update(f.t, f.obs);
      const want = seq.expected[i];
      assert.deepEqual(Object.keys(got).sort(), Object.keys(want).sort(), `keys @${i}`);
      for (const [k, w] of Object.entries(want)) {
        if (typeof w === 'number' && !Number.isInteger(w)) close(got[k], w, `${k} @${i} t=${f.t}`);
        else assert.equal(got[k], w, `${k} @${i} t=${f.t}`);
      }
    }
  });
}
