// 경보 구간 묶기: 시연 재생을 넣으면 자막대로 4건(하품·옆 보기·겹침·눈 감김)이 나와야 한다.
import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.location = { hostname: 'example.test' };   // config.js가 읽는다(저장소 꺼짐)
const { Monitor } = await import('../decision.js');
const { SCENARIO_SEC, observe } = await import('../scenario.js');
const { EpisodeRecorder, enabled } = await import('../store.js');

test('저장소 주소가 없으면 기록이 꺼진다', () => assert.equal(enabled, false));

test('시연 재생 → 이벤트 4건', () => {
  const mon = new Monitor();
  const rec = new EpisodeRecorder();
  const events = [];
  for (let t = 0; t < SCENARIO_SEC; t += 1 / 30) {
    const ev = rec.update(t, mon.update(t, observe(t)));
    if (ev) events.push(ev);
  }
  assert.deepEqual(events.map((e) => [e.level, e.cause]),
    [[1, 'yawn'], [1, 'distracted'], [2, 'compound'], [3, 'eyes_closed']]);
  const last = events[3];
  assert.ok(last.closed_sec > 2.4 && last.closed_sec < 3.0, `closed_sec ${last.closed_sec}`);
  assert.ok(last.duration_ms > 400 && last.duration_ms < 1100, `duration ${last.duration_ms}`);
});

test('0.4초보다 짧은 신호는 기록하지 않는다', () => {
  const rec = new EpisodeRecorder();
  const st = (level) => ({ calibrating: false, face_detected: true, alert_level: level, is_yawning: level > 0,
    is_distracted: false, is_drowsy: false, closed_sec: 0, perclos: 0, yaw: 0, pitch: 0 });
  assert.equal(rec.update(0.0, st(1)), null);
  assert.equal(rec.update(0.2, st(1)), null);
  assert.equal(rec.update(0.3, st(0)), null);
});
