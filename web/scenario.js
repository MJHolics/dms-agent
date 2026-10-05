// scenario.js — 카메라 없이 도는 시연용 합성 신호. 판단은 카메라 모드와 같은 Monitor가 한다.

export const SCENARIO_SEC = 30;

// [시작, 끝, 종류, 자막]
export const EVENTS = [
  [0.0, 2.2, 'calib', '운전을 시작하면 2초 동안 기준 자세를 잽니다'],
  [2.2, 5.0, 'normal', '정상 주행'],
  [5.0, 5.18, 'closed', '깜빡임 — 경보를 내지 않습니다'],
  [5.18, 7.5, 'normal', '깜빡임 — 경보를 내지 않습니다'],
  [7.5, 10.0, 'yawn', '하품 — 주의'],
  [10.0, 11.5, 'normal', '정상 주행'],
  [11.5, 14.0, 'turn', '옆을 봄 — 주의'],
  [14.0, 15.5, 'normal', '정상 주행'],
  [15.5, 18.5, 'yawn+turn', '하품하며 옆을 봄 — 신호 두 개가 겹치면 위험'],
  [18.5, 21.0, 'normal', '정상 주행'],
  [21.0, 23.8, 'closed', '눈을 감음 — 2초가 지나면 경보'],
  [23.8, 30.0, 'normal', '눈을 뜨면 경보가 풀립니다'],
];

// 결정적 잡음(재생할 때마다 같은 그림).
const wobble = (t, f, p) => Math.sin(t * f + p);
const ease = (x) => x * x * (3 - 2 * x);
// 구간 [a, b] 안에서 0→1→0, 양 끝 ramp초 동안 부드럽게.
function envelope(t, a, b, ramp) {
  if (t <= a || t >= b) return 0;
  return ease(Math.min(1, (t - a) / ramp, (b - t) / ramp));
}

export function eventAt(t) {
  return EVENTS.find(([a, b]) => t >= a && t < b) || EVENTS[EVENTS.length - 1];
}

// t(초, 0~SCENARIO_SEC) → 관측 한 건
export function observe(t) {
  let earV = 0.31 + 0.008 * wobble(t, 3.1, 0.4);
  let marV = 0.08 + 0.02 * wobble(t, 1.7, 1.1);
  let yaw = 2 + 2.5 * wobble(t, 0.9, 0.2);
  let pitch = 6 + 1.5 * wobble(t, 1.3, 2.0);

  for (const [a, b, kind] of EVENTS) {
    if (kind.includes('closed')) earV -= 0.23 * envelope(t, a, b, 0.07);
    if (kind.includes('yawn')) marV += 0.85 * envelope(t, a, b, 0.5);
    if (kind.includes('turn')) yaw += 46 * envelope(t, a, b, 0.4);
  }
  return { face_detected: true, ear: earV, mar: marV, pitch, yaw, detected_objects: [] };
}
