// store.js — 경보 한 번(시작~해제)을 이벤트 한 건으로 묶어 저장소에 보낸다. 영상·랜드마크는 보내지 않는다.
import { API_URL, API_KEY } from './config.js';
import { CLOSED_SEC_THRESH } from './decision.js';

export const enabled = Boolean(API_URL);
const MIN_EPISODE_SEC = 0.4;      // 이보다 짧게 스친 신호는 기록하지 않는다

function headers(extra = {}) {
  const h = { 'Content-Type': 'application/json', ...extra };
  if (API_KEY) { h.apikey = API_KEY; h.Authorization = `Bearer ${API_KEY}`; }
  return h;
}

export async function rest(path, { method = 'GET', body } = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    method, headers: headers(method === 'GET' ? {} : { Prefer: 'return=minimal' }),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return method === 'GET' ? res.json() : null;
}

// 판단 한 건 → 사유 코드(db/schema.sql의 cause와 같은 값)
export function causeOf(st) {
  if (!st.face_detected) return 'no_face';
  if (st.is_drowsy) return st.closed_sec >= CLOSED_SEC_THRESH ? 'eyes_closed' : 'perclos';
  if (st.is_yawning && st.is_distracted) return 'compound';
  if (st.is_yawning) return 'yawn';
  return 'distracted';
}

// 경보 구간 묶기. update()는 구간이 끝난 프레임에서만 이벤트를 돌려준다(그 밖엔 null).
export class EpisodeRecorder {
  constructor() { this.cur = null; }

  update(t, st) {
    const level = st.calibrating ? 0 : st.alert_level;
    if (level > 0) {
      if (!this.cur) this.cur = { t0: t, t1: t, level: 0, cause: null, closed_sec: 0, perclos: 0, yaw: 0, pitch: 0 };
      const c = this.cur;
      c.t1 = t;
      if (level > c.level) { c.level = level; c.cause = causeOf(st); c.yaw = st.yaw ?? 0; c.pitch = st.pitch ?? 0; }
      c.closed_sec = Math.max(c.closed_sec, st.closed_sec);
      c.perclos = Math.max(c.perclos, st.perclos);
      return null;
    }
    const c = this.cur;
    this.cur = null;
    if (!c || c.t1 - c.t0 < MIN_EPISODE_SEC) return null;
    const r2 = (v) => Math.round(v * 100) / 100;
    return {
      duration_ms: Math.round((c.t1 - c.t0) * 1000), level: c.level, cause: c.cause,
      closed_sec: r2(c.closed_sec), perclos: r2(c.perclos), yaw: r2(c.yaw), pitch: r2(c.pitch),
    };
  }
}

// 세션 하나(운행 한 번). 실패해도 화면은 계속 돌아야 하므로 예외를 밖으로 내지 않는다.
export class Session {
  constructor() {
    this.id = crypto.randomUUID();
    this.vehicle = `체험 ${String(Math.floor(Math.random() * 900) + 100)}`;
    this.saved = 0;
    this.failed = 0;
    this.ready = enabled
      ? rest('/dms_sessions', { method: 'POST', body: { id: this.id, vehicle: this.vehicle } }).then(() => true, () => false)
      : Promise.resolve(false);
  }

  async log(event) {
    if (!(await this.ready)) { this.failed += 1; return false; }
    try {
      await rest('/dms_events', { method: 'POST', body: { session_id: this.id, ...event } });
      this.saved += 1;
      return true;
    } catch (e) {
      this.failed += 1;
      return false;
    }
  }
}
