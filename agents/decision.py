"""
decision.py — 판단 계층(순수 함수). 모델·OpenCV 없이 표준 라이브러리만 쓴다.

파이프라인(agents/pipeline.py)과 브라우저 이식본(web/decision.js)이 같은 규칙을 쓴다.
web/tests/parity.test.mjs가 이 파일의 출력을 정답으로 삼아 JS 쪽을 대조한다.
규칙을 바꾸면 tools/export_parity_fixture.py를 다시 돌려야 한다.
"""
import math

# ── 랜드마크 인덱스(MediaPipe Face Mesh 478점) ──────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# 입 안쪽 윤곽: [왼쪽 끝, 위 3점(왼→오), 오른쪽 끝, 아래 3점(오→왼)]
MOUTH     = [78, 81, 13, 311, 308, 402, 14, 178]

# ── 임계 ────────────────────────────────────────────────────────
EAR_THRESH     = 0.25
MAR_THRESH     = 0.60
PERCLOS_THRESH = 0.15
YAW_THRESH     = 30.0     # 좌우, 절댓값(도)
PITCH_THRESH   = 20.0     # 숙임이 양수(도)
CLOSED_SEC_THRESH = 2.0   # 연속 눈 감김(초)
PERCLOS_WINDOW_SEC = 60.0   # 1분 창. 30초로 두면 2~3초 감김 한두 번에 15%를 넘어 경보가 30초 가까이 안 풀렸다(실기기 확인, 10-05)

REASONS = {0: '정상', 1: '주의 필요', 2: '복합 위험 신호', 3: '심각한 졸음 운전'}


def ear(p):
    """눈 6점 → Eye Aspect Ratio. p = [(x, y), ...] 픽셀 좌표."""
    a = math.dist(p[1], p[5])
    b = math.dist(p[2], p[4])
    c = math.dist(p[0], p[3])
    return (a + b) / (2 * c)


def mar(p):
    """입 안쪽 8점 → Mouth Aspect Ratio. 다물면 0에 가깝다."""
    a = math.dist(p[1], p[7])
    b = math.dist(p[2], p[6])
    c = math.dist(p[3], p[5])
    d = math.dist(p[0], p[4])
    return (a + b + c) / (2 * d)


def head_angles(m):
    """얼굴 변환 행렬(4x4, 행 우선 중첩 리스트) → (pitch, yaw) 도.

    얼굴 정면 벡터 = 회전부의 셋째 열. 카메라 좌표는 x 오른쪽·y 위·z 카메라 쪽.
    pitch는 숙일 때 양수, yaw는 부호 없이 절댓값만 판단에 쓴다.
    """
    fx, fy, fz = m[0][2], m[1][2], m[2][2]
    yaw = math.degrees(math.atan2(fx, fz))
    pitch = math.degrees(math.atan2(-fy, math.hypot(fx, fz)))
    return pitch, yaw


def classify(s):
    """신호 → 위험 플래그. s: face_detected, perclos, closed_sec, mar, yaw, pitch, detected_objects."""
    if not s.get('face_detected', True):
        return {'is_drowsy': False, 'is_yawning': False, 'is_distracted': False,
                'has_danger_obj': False, 'risk_count': 1}

    # 졸음은 '지속'이다 — 순간 깜빡임(단발 EAR 하강)을 졸음으로 오판하면 오경보가 난다.
    # tools/bench_fusion.py로 brief_blink 오경보 50%를 발견해 순간 EAR 조건을 뺐다.
    # 누적(PERCLOS) 또는 연속 감김 시간으로만 판정한다.
    d = ((s.get('perclos') or 0) > PERCLOS_THRESH
         or (s.get('closed_sec') or 0) >= CLOSED_SEC_THRESH)
    y = (s.get('mar') or 0) > MAR_THRESH
    i = abs(s.get('yaw') or 0) > YAW_THRESH or (s.get('pitch') or 0) > PITCH_THRESH
    o = len(s.get('detected_objects') or []) > 0

    return {'is_drowsy': d, 'is_yawning': y, 'is_distracted': i,
            'has_danger_obj': o, 'risk_count': sum([d, y, i, o])}


def alert(flags, detected_objects=None):
    """위험 플래그 → (경보 레벨 0~3, 사유)."""
    r = flags['risk_count']
    if flags['has_danger_obj']:
        return 3, f"위험 물체: {detected_objects[0]['class']}"
    if flags['is_drowsy'] or r >= 3:
        return 3, REASONS[3]
    if r == 2:
        return 2, REASONS[2]
    if r == 1:
        return 1, REASONS[1]
    return 0, REASONS[0]


class DrowsinessTracker:
    """EAR 시계열 → PERCLOS·연속 감김 시간. 프레임 수가 아니라 시각(초)으로 센다.

    프레임 수 창은 fps에 따라 길이가 달라지고, 창이 차기 전에는 깜빡임 한 번이 비율을 넘긴다.
    여기서는 표본마다 직전 표본과의 간격만큼 시간을 더하고, 창 길이로 나눈다
    (아직 안 본 구간은 뜬 눈으로 친다).
    """

    def __init__(self, window_sec=PERCLOS_WINDOW_SEC, ear_thresh=EAR_THRESH, max_gap_sec=0.5):
        self.window_sec = window_sec
        self.ear_thresh = ear_thresh
        self.max_gap_sec = max_gap_sec
        self.samples = []          # (t, dt, closed)
        self.closed_sum = 0.0
        self.closed_sec = 0.0
        self.last_t = None

    def update(self, t, ear_value):
        """t: 초. ear_value가 None이면(얼굴 없음) 연속 감김만 끊고 시간은 흘려보낸다."""
        dt = 0.0 if self.last_t is None else min(max(t - self.last_t, 0.0), self.max_gap_sec)
        self.last_t = t
        closed = ear_value is not None and ear_value < self.ear_thresh
        self.closed_sec = self.closed_sec + dt if closed else 0.0
        self.samples.append((t, dt, closed))
        if closed:
            self.closed_sum += dt
        while self.samples and self.samples[0][0] <= t - self.window_sec:
            _, odt, oclosed = self.samples.pop(0)
            if oclosed:
                self.closed_sum -= odt
        perclos = min(1.0, max(0.0, self.closed_sum / self.window_sec))
        return {'perclos': perclos, 'closed_sec': self.closed_sec}


# ── 기준 자세 보정 ──────────────────────────────────────────────
CALIB_SEC = 2.0
CALIB_MIN_SAMPLES = 10
CALIB_EAR_RATIO = 0.65            # 뜬 눈 EAR의 이 비율 밑을 감김으로 본다(0.75는 가늘게 뜬 눈까지 감김으로 셌다)
CALIB_EAR_RANGE = (0.15, 0.30)


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


class Calibrator:
    """처음 몇 초의 중앙값을 그 사람·그 카메라 위치의 기준으로 삼는다.

    카메라가 얼굴 정면에 있지 않으면(대시보드 구석, 노트북 위쪽) 정면을 봐도 각도가 0이 아니고,
    뜬 눈의 EAR도 사람마다 다르다. 고정 임계만 쓰면 가만히 있어도 경보가 난다.
    """

    def __init__(self, duration_sec=CALIB_SEC, min_samples=CALIB_MIN_SAMPLES):
        self.duration_sec = duration_sec
        self.min_samples = min_samples
        self.t0 = None
        self.ears, self.pitches, self.yaws = [], [], []
        self.done = False
        self.ear_thresh = EAR_THRESH
        self.pitch0 = 0.0
        self.yaw0 = 0.0

    def update(self, t, ear_value, pitch, yaw):
        """얼굴이 보이는 표본만 넣는다. 보정이 끝나면 True."""
        if self.done:
            return True
        if self.t0 is None:
            self.t0 = t
        self.ears.append(ear_value)
        self.pitches.append(pitch)
        self.yaws.append(yaw)
        if t - self.t0 >= self.duration_sec and len(self.ears) >= self.min_samples:
            lo, hi = CALIB_EAR_RANGE
            self.ear_thresh = min(hi, max(lo, _median(self.ears) * CALIB_EAR_RATIO))
            self.pitch0 = _median(self.pitches)
            self.yaw0 = _median(self.yaws)
            self.done = True
        return self.done


class Monitor:
    """관측 한 건(시각 + 신호) → 판단 한 건. 보정 → 시간 누적 → 분류 → 경보."""

    def __init__(self, calibrate=True):
        self.calibrator = Calibrator() if calibrate else None
        self.tracker = DrowsinessTracker()

    def update(self, t, obs):
        """obs: face_detected, ear, mar, pitch, yaw, detected_objects. t는 초."""
        face = bool(obs.get('face_detected'))
        objs = obs.get('detected_objects') or []
        cal = self.calibrator

        if cal is not None and not cal.done:
            if face:
                cal.update(t, obs['ear'], obs['pitch'], obs['yaw'])
                if cal.done:
                    self.tracker.ear_thresh = cal.ear_thresh
            return {'calibrating': True, 'face_detected': face, 'ear_thresh': self.tracker.ear_thresh,
                    'perclos': 0.0, 'closed_sec': 0.0, 'pitch': None, 'yaw': None,
                    'is_drowsy': False, 'is_yawning': False, 'is_distracted': False,
                    'has_danger_obj': False, 'risk_count': 0, 'alert_level': 0, 'alert_reason': '기준 자세 측정 중'}

        acc = self.tracker.update(t, obs.get('ear') if face else None)
        pitch = obs['pitch'] - (cal.pitch0 if cal else 0.0) if face else None
        yaw = obs['yaw'] - (cal.yaw0 if cal else 0.0) if face else None
        s = {'face_detected': face, 'perclos': acc['perclos'], 'closed_sec': acc['closed_sec'],
             'mar': obs.get('mar') if face else None, 'pitch': pitch, 'yaw': yaw, 'detected_objects': objs}
        flags = classify(s)
        level, reason = alert(flags, objs)
        if not face:
            reason = '얼굴이 보이지 않음'
        return {'calibrating': False, 'face_detected': face, 'ear_thresh': self.tracker.ear_thresh,
                'perclos': acc['perclos'], 'closed_sec': acc['closed_sec'], 'pitch': pitch, 'yaw': yaw,
                **flags, 'alert_level': level, 'alert_reason': reason}
