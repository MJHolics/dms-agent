// decision.js — agents/decision.py를 옮긴 것. 원본은 파이썬이다.
// 규칙을 바꾸면 파이썬을 먼저 고치고 tools/export_parity_fixture.py → `node --test web/tests`로 대조한다.

export const LEFT_EYE = [362, 385, 387, 263, 373, 380];
export const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
export const MOUTH = [78, 81, 13, 311, 308, 402, 14, 178];

export const EAR_THRESH = 0.25;
export const MAR_THRESH = 0.60;
export const PERCLOS_THRESH = 0.60;
export const YAW_THRESH = 30.0;
export const PITCH_THRESH = 20.0;
export const CLOSED_SEC_THRESH = 2.0;
export const PERCLOS_WINDOW_SEC = 5.0;

export const REASONS = { 0: '정상', 1: '주의 필요', 2: '복합 위험 신호', 3: '심각한 졸음 운전' };

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const deg = (r) => (r * 180) / Math.PI;

export function ear(p) {
  const a = dist(p[1], p[5]);
  const b = dist(p[2], p[4]);
  const c = dist(p[0], p[3]);
  return (a + b) / (2 * c);
}

export function mar(p) {
  const a = dist(p[1], p[7]);
  const b = dist(p[2], p[6]);
  const c = dist(p[3], p[5]);
  const d = dist(p[0], p[4]);
  return (a + b + c) / (2 * d);
}

// m: 4x4 행 우선 중첩 배열. 반환 [pitch, yaw](도), 숙임이 양수.
export function headAngles(m) {
  const fx = m[0][2], fy = m[1][2], fz = m[2][2];
  const yaw = deg(Math.atan2(fx, fz));
  const pitch = deg(Math.atan2(-fy, Math.hypot(fx, fz)));
  return [pitch, yaw];
}

export function classify(s) {
  if (s.face_detected === false) {
    return { is_drowsy: false, is_yawning: false, is_distracted: false, has_danger_obj: false, risk_count: 1 };
  }
  const d = (s.perclos || 0) > PERCLOS_THRESH || (s.closed_sec || 0) >= CLOSED_SEC_THRESH;
  const y = (s.mar || 0) > MAR_THRESH;
  const i = Math.abs(s.yaw || 0) > YAW_THRESH || (s.pitch || 0) > PITCH_THRESH;
  const o = (s.detected_objects || []).length > 0;
  return { is_drowsy: d, is_yawning: y, is_distracted: i, has_danger_obj: o, risk_count: d + y + i + o };
}

export function alert(flags, detectedObjects) {
  const r = flags.risk_count;
  if (flags.has_danger_obj) return [3, `위험 물체: ${detectedObjects[0].class}`];
  if (flags.is_drowsy || r >= 3) return [3, REASONS[3]];
  if (r === 2) return [2, REASONS[2]];
  if (r === 1) return [1, REASONS[1]];
  return [0, REASONS[0]];
}

export class DrowsinessTracker {
  constructor(windowSec = PERCLOS_WINDOW_SEC, earThresh = EAR_THRESH, maxGapSec = 0.5) {
    this.windowSec = windowSec;
    this.earThresh = earThresh;
    this.maxGapSec = maxGapSec;
    this.samples = [];
    this.closedSum = 0.0;
    this.closedSec = 0.0;
    this.lastT = null;
  }

  update(t, earValue) {
    const dt = this.lastT === null ? 0.0 : Math.min(Math.max(t - this.lastT, 0.0), this.maxGapSec);
    this.lastT = t;
    const closed = earValue !== null && earValue !== undefined && earValue < this.earThresh;
    this.closedSec = closed ? this.closedSec + dt : 0.0;
    this.samples.push([t, dt, closed]);
    if (closed) this.closedSum += dt;
    while (this.samples.length && this.samples[0][0] <= t - this.windowSec) {
      const [, odt, oclosed] = this.samples.shift();
      if (oclosed) this.closedSum -= odt;
    }
    const perclos = Math.min(1.0, Math.max(0.0, this.closedSum / this.windowSec));
    return { perclos, closed_sec: this.closedSec };
  }
}

export const CALIB_SEC = 2.0;
export const CALIB_MIN_SAMPLES = 10;
export const CALIB_EAR_RATIO = 0.65;
export const CALIB_EAR_RANGE = [0.15, 0.30];

export const BLINK_OPEN_THRESH = 0.40;
export const BLINK_OPEN_RATIO = 0.45;
export const BLINK_OPEN_RANGE = [0.30, 0.50];

export const SQUINT_RELIEF = 0.40;

export function squintScore(eyeSquint, cheekSquint, mouthSmile) {
  return Math.max(eyeSquint, cheekSquint, mouthSmile);
}

export function eyeOpenness(blinkLeft, blinkRight, squint = 0.0) {
  return Math.min(1.0, 1.0 - Math.min(blinkLeft, blinkRight) + SQUINT_RELIEF * squint);
}

function median(v) {
  const s = [...v].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

export class Calibrator {
  constructor(durationSec = CALIB_SEC, minSamples = CALIB_MIN_SAMPLES,
    earRatio = CALIB_EAR_RATIO, earRange = CALIB_EAR_RANGE, earThresh = EAR_THRESH) {
    this.durationSec = durationSec;
    this.minSamples = minSamples;
    this.earRatio = earRatio;
    this.earRange = earRange;
    this.t0 = null;
    this.ears = [];
    this.pitches = [];
    this.yaws = [];
    this.done = false;
    this.earThresh = earThresh;
    this.pitch0 = 0.0;
    this.yaw0 = 0.0;
  }

  update(t, earValue, pitch, yaw) {
    if (this.done) return true;
    if (this.t0 === null) this.t0 = t;
    this.ears.push(earValue);
    this.pitches.push(pitch);
    this.yaws.push(yaw);
    if (t - this.t0 >= this.durationSec && this.ears.length >= this.minSamples) {
      const [lo, hi] = this.earRange;
      this.earThresh = Math.min(hi, Math.max(lo, median(this.ears) * this.earRatio));
      this.pitch0 = median(this.pitches);
      this.yaw0 = median(this.yaws);
      this.done = true;
    }
    return this.done;
  }

  // 0~1, 화면의 진행 표시용(판단에는 쓰지 않는다).
  progress(t) {
    if (this.done) return 1;
    if (this.t0 === null) return 0;
    return Math.min(1, (t - this.t0) / this.durationSec);
  }
}

export class Monitor {
  constructor(calibrate = true, earRatio = CALIB_EAR_RATIO, earRange = CALIB_EAR_RANGE, earThresh = EAR_THRESH) {
    this.calibrator = calibrate ? new Calibrator(CALIB_SEC, CALIB_MIN_SAMPLES, earRatio, earRange, earThresh) : null;
    this.tracker = new DrowsinessTracker(PERCLOS_WINDOW_SEC, earThresh);
  }

  // obs: face_detected, ear, mar, pitch, yaw, detected_objects. t는 초.
  update(t, obs) {
    const face = Boolean(obs.face_detected);
    const objs = obs.detected_objects || [];
    const cal = this.calibrator;

    if (cal !== null && !cal.done) {
      if (face) {
        cal.update(t, obs.ear, obs.pitch, obs.yaw);
        if (cal.done) this.tracker.earThresh = cal.earThresh;
      }
      return {
        calibrating: true, face_detected: face, ear_thresh: this.tracker.earThresh,
        perclos: 0.0, closed_sec: 0.0, pitch: null, yaw: null,
        is_drowsy: false, is_yawning: false, is_distracted: false,
        has_danger_obj: false, risk_count: 0, alert_level: 0, alert_reason: '기준 자세 측정 중',
      };
    }

    const acc = this.tracker.update(t, face ? obs.ear : null);
    const pitch = face ? obs.pitch - (cal ? cal.pitch0 : 0.0) : null;
    const yaw = face ? obs.yaw - (cal ? cal.yaw0 : 0.0) : null;
    const s = {
      face_detected: face, perclos: acc.perclos, closed_sec: acc.closed_sec,
      mar: face ? obs.mar : null, pitch, yaw, detected_objects: objs,
    };
    const flags = classify(s);
    let [level, reason] = alert(flags, objs);
    if (!face) reason = '얼굴이 보이지 않음';
    return {
      calibrating: false, face_detected: face, ear_thresh: this.tracker.earThresh,
      perclos: acc.perclos, closed_sec: acc.closed_sec, pitch, yaw,
      ...flags, alert_level: level, alert_reason: reason,
    };
  }
}
