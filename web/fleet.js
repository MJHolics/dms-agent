// fleet.js — 관제 화면. db/schema.sql의 읽기용 뷰 세 개만 조회한다(테이블은 익명에게 닫혀 있다).
import { enabled, rest } from './store.js';

const $ = (id) => document.getElementById(id);
const CAUSE = {
  eyes_closed: '눈 감김 2초 이상', perclos: '눈 감은 비율 초과', yawn: '하품',
  distracted: '앞을 보지 않음', no_face: '얼굴 안 보임', compound: '신호 두 개 겹침',
};
const LEVEL = { 1: '주의', 2: '위험', 3: '경보' };
const COLOR = { 1: '#f2c14e', 2: '#f08a3c', 3: '#ff3b4e' };
const REFRESH_SEC = 5;
const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ESC[c]);
const when = (iso) => new Date(iso).toLocaleString('ko-KR', {
  month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
});

function drawHourly(rows) {
  const cv = $('hourly'), ctx = cv.getContext('2d'), w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  const hours = 7 * 24, now = Date.now(), slot = new Map();
  for (const r of rows) {
    const i = hours - 1 - Math.floor((now - new Date(r.hour).getTime()) / 3600e3);
    if (i < 0 || i >= hours) continue;
    const s = slot.get(i) || { 1: 0, 2: 0, 3: 0 };
    s[r.level] += r.alerts;
    slot.set(i, s);
  }
  const max = Math.max(1, ...[...slot.values()].map((s) => s[1] + s[2] + s[3]));
  const bw = w / hours;
  for (const [i, s] of slot) {
    let y = h - 30;
    for (const lv of [1, 2, 3]) {
      const bh = (s[lv] / max) * (h - 60);
      ctx.fillStyle = COLOR[lv];
      ctx.fillRect(i * bw + 1, y - bh, Math.max(1, bw - 2), bh);
      y -= bh;
    }
  }
  ctx.fillStyle = '#8b9bb0';
  ctx.font = '22px sans-serif';
  for (let d = 0; d < 7; d++) {
    const t = new Date(now - (6 - d) * 86400e3);
    ctx.fillText(`${t.getMonth() + 1}/${t.getDate()}`, (d + 0.4) * 24 * bw, h - 4);
  }
  ctx.fillText(`가장 많은 시간 ${max}건`, 6, 22);
}

async function refresh() {
  try {
    const t0 = performance.now();
    const [recent, vehicles, hourly] = await Promise.all([
      rest('/dms_recent_events?limit=12'), rest('/dms_vehicle_summary'), rest('/dms_hourly'),
    ]);
    const ms = performance.now() - t0;

    const sum = (k) => vehicles.reduce((a, v) => a + v[k], 0);
    $('tAll').textContent = sum('alerts_24h').toLocaleString();
    $('tCrit').textContent = sum('critical_24h').toLocaleString();
    $('tDrowsy').textContent = sum('drowsy_24h').toLocaleString();
    $('tVeh').textContent = vehicles.length;

    $('vehicles').innerHTML = vehicles.slice(0, 10).map((v) => `<tr><td>${esc(v.vehicle)}</td>
      <td class="num">${v.alerts_24h}</td><td class="num">${v.critical_24h}</td><td class="num">${v.drowsy_24h}</td>
      <td>${when(v.last_alert_at)}</td></tr>`).join('')
      || '<tr><td colspan="5">24시간 안에 들어온 경보가 없습니다.</td></tr>';
    $('recent').innerHTML = recent.map((e) => `<tr><td>${when(e.ended_at)}</td>
      <td class="${e.source === 'seed' ? 'seed' : ''}">${esc(e.vehicle)}</td>
      <td><span class="lv lv${e.level}">${LEVEL[e.level]}</span></td>
      <td>${CAUSE[e.cause] || esc(e.cause)}</td><td class="num">${(e.duration_ms / 1000).toFixed(1)}초</td></tr>`).join('')
      || '<tr><td colspan="5">아직 기록이 없습니다.</td></tr>';
    drawHourly(hourly);
    $('status').textContent = `${REFRESH_SEC}초마다 새로 읽습니다 · 방금 조회 3건에 ${ms.toFixed(0)}ms · `
      + '흐린 글씨의 차량은 화면을 채우려고 넣은 예시 기록이고, "체험 NNN"은 방문자가 방금 남긴 기록입니다. 영상은 저장하지 않습니다.';
  } catch (e) {
    $('status').textContent = `저장소에서 읽지 못했습니다: ${e.message}`;
  }
}

if (!enabled) {
  $('status').textContent = '이 주소에는 저장소가 연결돼 있지 않습니다. 저장소(db/schema.sql)를 띄우고 config.js에 주소를 넣으면 켜집니다.';
} else {
  refresh();
  setInterval(() => { if (!document.hidden) refresh(); }, REFRESH_SEC * 1000);
  window.__refresh = refresh;
}
