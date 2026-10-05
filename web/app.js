import * as D from './decision.js';
import { SCENARIO_SEC, eventAt, observe } from './scenario.js';
import * as store from './store.js';

const TASKS_VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL_URLS = [
  './models/face_landmarker.task',
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
];
const ALARM_HOLD_SEC = 2.5;   // 눈을 뜬 뒤에도 경보를 이만큼 유지한다(화면·소리만, 판단값은 그대로)
const HISTORY_SEC = 30;
const LEVEL_NAME = ['정상', '주의', '위험', '경보'];
const LEVEL_COLOR = ['#2fbf71', '#f2c14e', '#f08a3c', '#ff3b4e'];

const $ = (id) => document.getElementById(id);
const view = $('view');
const ctx = view.getContext('2d');
const tl = $('timeline');
const tctx = tl.getContext('2d');

let mode = 'demo';            // 'demo' | 'camera'
let monitor = new D.Monitor();
let history = [];             // {t, ear, level}
let lastL3 = -Infinity;
let demoStart = performance.now() / 1000;
let landmarker = null;
let stream = null;
const video = document.createElement('video');
video.playsInline = true;
video.muted = true;

// ── 경보음 ────────────────────────────────────────────────────
let audio = null;
let soundOn = false;
let nextBeep = 0;
function beep(now) {
  if (!soundOn || !audio || now < nextBeep) return;
  nextBeep = now + 0.32;
  const o = audio.createOscillator();
  const g = audio.createGain();
  o.type = 'square';
  o.frequency.value = 1320;
  g.gain.setValueAtTime(0.0001, audio.currentTime);
  g.gain.exponentialRampToValueAtTime(0.25, audio.currentTime + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.2);
  o.connect(g).connect(audio.destination);
  o.start();
  o.stop(audio.currentTime + 0.22);
}
function enableSound(on) {
  soundOn = on;
  if (on && !audio) audio = new (window.AudioContext || window.webkitAudioContext)();
  if (on && audio.state === 'suspended') audio.resume();
  $('btnSound').textContent = on ? '경보음 끄기' : '경보음 켜기';
}
$('btnSound').addEventListener('click', () => enableSound(!soundOn));

// ── 화면 갱신 ─────────────────────────────────────────────────
function setBar(id, ratio, hot) {
  const el = $(id);
  el.style.width = `${Math.max(0, Math.min(1, ratio)) * 100}%`;
  el.classList.toggle('hot', Boolean(hot));
}

// 경보 사유를 화면용 문장으로. 판단값(st)은 그대로 두고 표현만 바꾼다.
function explain(st) {
  if (!st.face_detected) return '얼굴이 보이지 않습니다';
  if (st.is_drowsy) {
    return st.closed_sec >= D.CLOSED_SEC_THRESH
      ? `눈을 ${st.closed_sec.toFixed(1)}초째 감고 있습니다`
      : `최근 30초의 ${(st.perclos * 100).toFixed(0)}%를 눈 감고 있었습니다`;
  }
  const why = [];
  if (st.is_yawning) why.push('하품');
  if (st.is_distracted) why.push('앞을 보지 않음');
  return why.length ? why.join(' + ') : st.alert_reason;
}

function render(now, obs, st) {
  if (st.alert_level === 3) lastL3 = now;
  const shown = now - lastL3 < ALARM_HOLD_SEC ? 3 : st.alert_level;
  if (shown === 3) {
    beep(now);
    if (st.alert_level === 3 && navigator.vibrate && soundOn) navigator.vibrate(120);
  }

  $('stage').dataset.level = st.calibrating ? 0 : shown;
  $('lv').textContent = st.calibrating ? '기준 측정' : LEVEL_NAME[shown];
  $('why').textContent = st.calibrating
    ? '정면을 봐 주세요'
    : (shown === 3 && st.alert_level !== 3 ? '경보 유지 중' : (shown === 0 ? '' : explain(st)));

  $('cDrowsy').classList.toggle('on', st.is_drowsy);
  $('cYawn').classList.toggle('on', st.is_yawning);
  $('cDistract').classList.toggle('on', st.is_distracted || (!st.face_detected && !st.calibrating));

  $('vClosed').textContent = `${st.closed_sec.toFixed(1)}초 / ${D.CLOSED_SEC_THRESH}초`;
  setBar('bClosed', st.closed_sec / D.CLOSED_SEC_THRESH, st.closed_sec >= D.CLOSED_SEC_THRESH);
  $('vPerclos').textContent = `${(st.perclos * 100).toFixed(0)}% / ${D.PERCLOS_THRESH * 100}%`;
  setBar('bPerclos', st.perclos / 0.4, st.perclos > D.PERCLOS_THRESH);

  const face = obs.face_detected;
  $('vEar').textContent = face ? `${obs.ear.toFixed(2)} (기준 ${st.ear_thresh.toFixed(2)})` : '얼굴 없음';
  setBar('bEar', face ? obs.ear / 0.45 : 0, face && obs.ear < st.ear_thresh);
  $('mEar').style.left = `${(st.ear_thresh / 0.45) * 100}%`;
  $('vMar').textContent = face ? obs.mar.toFixed(2) : '-';
  setBar('bMar', face ? obs.mar / 1.2 : 0, st.is_yawning);
  if (st.yaw === null || st.yaw === undefined) {
    $('vPose').textContent = '-';
    setBar('bPose', 0, false);
  } else {
    $('vPose').textContent = `${st.yaw.toFixed(0)}° / ${st.pitch.toFixed(0)}°`;
    const r = Math.max(Math.abs(st.yaw) / D.YAW_THRESH, st.pitch / D.PITCH_THRESH, 0);
    setBar('bPose', r / (1 / 0.6), st.is_distracted);
  }

  history.push({ t: now, ear: face ? obs.ear : null, level: st.calibrating ? 0 : st.alert_level });
  while (history.length && history[0].t < now - HISTORY_SEC) history.shift();
  drawTimeline(now, st.ear_thresh);
}

function drawTimeline(now, earThresh) {
  const w = tl.width, h = tl.height;
  tctx.clearRect(0, 0, w, h);
  const x = (t) => w - ((now - t) / HISTORY_SEC) * w;
  const y = (e) => h - 8 - (Math.min(e, 0.45) / 0.45) * (h - 16);
  for (let i = 0; i < history.length; i++) {
    const a = history[i];
    const end = i + 1 < history.length ? history[i + 1].t : now;
    tctx.fillStyle = LEVEL_COLOR[a.level];
    tctx.globalAlpha = a.level === 0 ? 0.18 : 0.75;
    tctx.fillRect(x(a.t), 0, Math.max(1, x(end) - x(a.t) + 0.5), h);
  }
  tctx.globalAlpha = 1;
  tctx.strokeStyle = 'rgba(255,255,255,.35)';
  tctx.setLineDash([4, 4]);
  tctx.beginPath(); tctx.moveTo(0, y(earThresh)); tctx.lineTo(w, y(earThresh)); tctx.stroke();
  tctx.setLineDash([]);
  tctx.strokeStyle = '#fff';
  tctx.lineWidth = 2;
  tctx.beginPath();
  let pen = false;
  for (const p of history) {
    if (p.ear === null) { pen = false; continue; }
    if (pen) tctx.lineTo(x(p.t), y(p.ear)); else tctx.moveTo(x(p.t), y(p.ear));
    pen = true;
  }
  tctx.stroke();
}

// ── 시연 재생(합성 신호 + 그림) ───────────────────────────────
function drawAvatar(obs, st) {
  const w = view.width, h = view.height;
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, '#16222e'); g.addColorStop(1, '#0a1016');
  ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
  // 운전석 창과 머리받이
  ctx.fillStyle = '#1d3145'; ctx.fillRect(w * 0.06, h * 0.08, w * 0.3, h * 0.42);
  ctx.fillStyle = '#0e151c'; ctx.fillRect(w * 0.36, h * 0.08, w * 0.03, h * 0.42);
  ctx.fillStyle = '#222c37';
  ctx.beginPath(); ctx.roundRect(w * 0.37, h * 0.1, w * 0.26, h * 0.34, 26); ctx.fill();

  const open = Math.max(0, Math.min(1, (obs.ear - 0.08) / 0.23));
  const mouth = Math.max(0, Math.min(1, obs.mar / 0.95));
  const yaw = (st.yaw ?? 0) / 60;       // -1~1
  const pitch = (st.pitch ?? 0) / 60;
  const cx = w * 0.5 + yaw * 26, cy = h * 0.44 + pitch * 30;
  const fx = cx + yaw * 58, fy = cy + pitch * 46;   // 이목구비는 머리보다 더 움직인다

  // 몸·목
  ctx.fillStyle = '#31506b';
  ctx.beginPath(); ctx.ellipse(w * 0.5, h * 1.04, w * 0.3, h * 0.3, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#d9a583'; ctx.fillRect(w * 0.5 - 30, cy + 90, 60, 60);
  // 머리
  ctx.fillStyle = '#e8b796';
  ctx.beginPath(); ctx.ellipse(cx, cy, 104, 128, yaw * 0.12, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#2a2f36';
  ctx.beginPath(); ctx.ellipse(cx - yaw * 20, cy - 74 - pitch * 26, 108, 68, 0, Math.PI, Math.PI * 2); ctx.fill();
  // 눈
  for (const side of [-1, 1]) {
    const ex = fx + side * 42 * (1 - Math.abs(yaw) * 0.25), ey = fy - 12;
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.ellipse(ex, ey, 20, 1.5 + 11 * open, 0, 0, Math.PI * 2); ctx.fill();
    if (open > 0.25) {
      ctx.fillStyle = '#1d2733';
      ctx.beginPath(); ctx.arc(ex + yaw * 6, ey, Math.min(8, 11 * open), 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = '#3a2c25'; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(ex - 22, ey - 24 + (1 - open) * 6); ctx.lineTo(ex + 22, ey - 24 + (1 - open) * 6); ctx.stroke();
  }
  // 코·입
  ctx.strokeStyle = '#b98363'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(fx, fy + 4); ctx.lineTo(fx + yaw * 10 - 6, fy + 34); ctx.lineTo(fx + 6, fy + 34); ctx.stroke();
  ctx.fillStyle = '#6d2f33';
  ctx.beginPath(); ctx.ellipse(fx, fy + 66, 26 - mouth * 4, 2.5 + 30 * mouth, 0, 0, Math.PI * 2); ctx.fill();
  // 운전대
  ctx.strokeStyle = '#0c1117'; ctx.lineWidth = 22;
  ctx.beginPath(); ctx.arc(w * 0.5, h * 1.22, w * 0.3, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke();
}

function demoFrame(now) {
  let t = now - demoStart;
  if (t >= SCENARIO_SEC) {         // 한 바퀴 돌면 처음부터
    demoStart = now; t = 0;
    monitor = new D.Monitor(); history = []; lastL3 = -Infinity;
  }
  const obs = observe(t);
  const st = monitor.update(now, obs);
  drawAvatar(obs, st);
  $('caption').textContent = eventAt(t)[3];
  render(now, obs, st);
}

// ── 카메라 ────────────────────────────────────────────────────
async function loadLandmarker() {
  if (landmarker) return landmarker;
  const { FaceLandmarker, FilesetResolver } = await import(`${TASKS_VISION}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${TASKS_VISION}/wasm`);
  let lastErr;
  for (const url of MODEL_URLS) {
    for (const delegate of ['GPU', 'CPU']) {
      try {
        landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: url, delegate },
          runningMode: 'VIDEO', numFaces: 1, outputFacialTransformationMatrixes: true,
        });
        return landmarker;
      } catch (e) { lastErr = e; }
    }
  }
  throw lastErr;
}

// MediaPipe 결과 → 관측 한 건. w, h는 원본 프레임 크기(픽셀).
export function toObservation(result, w, h) {
  const lm = result.faceLandmarks && result.faceLandmarks[0];
  if (!lm) return { obs: { face_detected: false, ear: null, mar: null, pitch: null, yaw: null, detected_objects: [] }, lm: null };
  const pts = (idx) => idx.map((i) => [lm[i].x * w, lm[i].y * h]);
  const earV = (D.ear(pts(D.LEFT_EYE)) + D.ear(pts(D.RIGHT_EYE))) / 2;
  const marV = D.mar(pts(D.MOUTH));
  let pitch = 0, yaw = 0;
  const mat = result.facialTransformationMatrixes && result.facialTransformationMatrixes[0];
  if (mat) {
    const d = mat.data;
    // 마지막 행이 (0,0,0,1)이면 열 우선 저장이다.
    const colMajor = Math.abs(d[3]) + Math.abs(d[7]) + Math.abs(d[11]) < 1e-6;
    const m = [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => (colMajor ? d[c * 4 + r] : d[r * 4 + c])));
    [pitch, yaw] = D.headAngles(m);
  }
  return { obs: { face_detected: true, ear: earV, mar: marV, pitch, yaw, detected_objects: [] }, lm };
}

function drawCamera(lm, st) {
  const w = view.width, h = view.height;
  ctx.save();
  ctx.translate(w, 0); ctx.scale(-1, 1);          // 거울처럼 보이게
  ctx.drawImage(video, 0, 0, w, h);
  if (lm) {
    ctx.fillStyle = 'rgba(91,192,255,.55)';
    for (const p of lm) ctx.fillRect(p.x * w - 1, p.y * h - 1, 2, 2);
    const poly = (idx, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.beginPath();
      idx.forEach((i, k) => (k ? ctx.lineTo(lm[i].x * w, lm[i].y * h) : ctx.moveTo(lm[i].x * w, lm[i].y * h)));
      ctx.closePath(); ctx.stroke();
    };
    const eyeColor = st.is_drowsy ? LEVEL_COLOR[3] : (st.closed_sec > 0 ? LEVEL_COLOR[1] : LEVEL_COLOR[0]);
    poly(D.LEFT_EYE, eyeColor); poly(D.RIGHT_EYE, eyeColor);
    poly(D.MOUTH, st.is_yawning ? LEVEL_COLOR[1] : LEVEL_COLOR[0]);
  }
  ctx.restore();
}

// ── 경보 기록(카메라 모드에서만) ─────────────────────────────
let session = null;
let recorder = null;
function showLog() {
  if (!store.enabled) return;
  $('log').hidden = false;
  $('logText').textContent = session
    ? `${session.vehicle} · 경보 ${session.saved}건 저장${session.failed ? ` · 실패 ${session.failed}건` : ''}`
    : '카메라를 켜면 경보가 날 때마다 기록(시각·종류·수치)이 저장됩니다. 영상은 저장하지 않습니다.';
}
function record(now, st) {
  if (!recorder) return;
  const ev = recorder.update(now, st);
  if (ev) session.log(ev).then(showLog);
}
showLog();

function cameraFrame(now) {
  if (video.readyState < 2 || !video.videoWidth) return;
  if (view.width !== video.videoWidth) { view.width = video.videoWidth; view.height = video.videoHeight; }
  const result = landmarker.detectForVideo(video, performance.now());
  const { obs, lm } = toObservation(result, video.videoWidth, video.videoHeight);
  const st = monitor.update(now, obs);
  record(now, st);
  drawCamera(lm, st);
  $('caption').textContent = st.calibrating
    ? '기준 자세를 재는 중입니다. 정면을 봐 주세요.'
    : '눈을 2초 감아 보세요. 하품을 하거나 고개를 돌려도 됩니다.';
  render(now, obs.face_detected ? obs : { ...obs, ear: 0, mar: 0 }, st);
}

// 휴대폰 화면이 꺼지지 않게(지원하는 브라우저에서만).
let wakeLock = null;
async function keepAwake(on) {
  try {
    if (on && navigator.wakeLock) wakeLock = await navigator.wakeLock.request('screen');
    if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; }
  } catch (e) { /* 거부돼도 동작에는 지장 없다 */ }
}
document.addEventListener('visibilitychange', () => { if (!document.hidden && mode === 'camera') keepAwake(true); });

function resetState() {
  monitor = new D.Monitor(); history = []; lastL3 = -Infinity;
}

async function startCamera() {
  const btn = $('btnCam');
  btn.disabled = true;
  $('note').classList.remove('error');
  enableSound(true);              // 버튼을 누른 김에 소리도 켠다(브라우저는 사용자 동작 뒤에만 소리를 허용한다)
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('이 브라우저는 카메라를 지원하지 않습니다.');
    btn.textContent = '카메라 여는 중…';
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    btn.textContent = '모델 불러오는 중…';
    await loadLandmarker();
    resetState();
    mode = 'camera';
    $('tag').textContent = '내 카메라 · 이 기기 안에서 처리';
    btn.hidden = true;
    $('btnDemo').hidden = false;
    $('btnRecal').hidden = false;
    keepAwake(true);
    if (store.enabled) { session = new store.Session(); recorder = new store.EpisodeRecorder(); showLog(); }
  } catch (e) {
    stopCamera();
    const denied = e && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
    const none = e && (e.name === 'NotFoundError' || e.name === 'OverconstrainedError');
    $('note').textContent = denied ? '카메라 권한이 거부됐습니다. 주소창의 카메라 아이콘에서 허용한 뒤 다시 눌러 주세요. 시연 재생은 계속 볼 수 있습니다.'
      : none ? '카메라를 찾지 못했습니다. 휴대폰에서 열어 보세요. 시연 재생은 계속 볼 수 있습니다.'
        : `카메라 모드를 시작하지 못했습니다: ${e && e.message ? e.message : e}`;
    $('note').classList.add('error');
  } finally {
    btn.disabled = false;
    if (mode !== 'camera') btn.textContent = '내 얼굴로 해 보기';
  }
}

function stopCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
}

function startDemo() {
  stopCamera();
  mode = 'demo';
  view.width = 640; view.height = 480;
  resetState();
  demoStart = performance.now() / 1000;
  $('tag').textContent = '시연 재생 · 합성 신호';
  $('btnCam').hidden = false;
  $('btnCam').textContent = '내 얼굴로 해 보기';
  $('btnDemo').hidden = true;
  $('btnRecal').hidden = true;
  keepAwake(false);
  session = null; recorder = null; showLog();
}

$('btnCam').addEventListener('click', startCamera);
$('btnDemo').addEventListener('click', startDemo);
$('btnRecal').addEventListener('click', resetState);

let manual = false;           // 점검용 수동 진행 중에는 자동 루프를 세운다
function loop() {
  const now = performance.now() / 1000;
  try {
    if (manual) { /* 수동 진행 */ } else if (mode === 'camera') cameraFrame(now); else demoFrame(now);
  } catch (e) {
    console.error(e);
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// ── 자가 점검: ?selftest=1 — 같은 사진을 파이썬 파이프라인에 넣었을 때의 값과 견준다 ──
if (new URLSearchParams(location.search).has('selftest')) {
  (async () => {
    const out = $('selftest');
    try {
      // 가려진 탭에서는 Image.decode()가 끝나지 않아 비트맵으로 읽는다.
      const img = await createImageBitmap(await (await fetch('./tests/face.png')).blob());
      const lmk = await loadLandmarker();
      await lmk.setOptions({ runningMode: 'IMAGE' });
      const { obs } = toObservation(lmk.detect(img), img.width, img.height);
      await lmk.setOptions({ runningMode: 'VIDEO' });
      const py = { ear: 0.3124, mar: 0.2213, pitch: 14.92, yaw: -0.29 };
      const rows = Object.keys(py).map((k) => `${k}: 브라우저 ${obs[k].toFixed(4)} · 파이썬 ${py[k]} · 차이 ${(obs[k] - py[k]).toFixed(4)}`);
      out.textContent = `자가 점검(tests/face.png)\n${rows.join('\n')}`;
      window.__selftest = { obs, py };
    } catch (e) {
      out.textContent = `자가 점검 실패: ${e && e.message ? e.message : e}`;
      window.__selftest = { error: String(e) };
    }
  })();
}

// 점검용: 가려진 탭에서는 requestAnimationFrame이 멈추므로 시각을 직접 넣어 한 프레임씩 돌릴 수 있게 둔다.
window.__dms = {
  get mode() { return mode; },
  get last() { return history[history.length - 1]; },
  restartDemo(now) { manual = true; resetState(); demoStart = now; },
  tick(now) { if (mode === 'camera') cameraFrame(now); else demoFrame(now); },
};
