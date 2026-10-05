"""
export_parity_fixture.py — 브라우저 이식본(web/decision.js) 대조용 정답 파일을 만든다.

파이썬 판단 계층(agents/decision.py)에 입력을 태워 입력과 출력을 함께 web/tests/fixture.json에 쓴다.
`node --test web/tests`가 같은 입력을 JS에 태워 이 출력과 비교한다.

    python tools/export_parity_fixture.py
"""
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents import decision as D  # noqa: E402
from tools.bench_fusion import make_dataset  # noqa: E402

SEED = 7


def _points(rng, n):
    return [[rng.uniform(0, 640), rng.uniform(0, 480)] for _ in range(n)]


def _rotation(rng):
    """yaw·pitch·roll을 무작위로 준 4x4 변환 행렬(행 우선)."""
    y, p, r = (math.radians(rng.uniform(-70, 70)) for _ in range(3))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]
    mul = lambda a, b: [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    m = mul(rz, mul(ry, rx))
    return [m[0] + [rng.uniform(-5, 5)], m[1] + [rng.uniform(-5, 5)], m[2] + [-40.0], [0, 0, 0, 1]]


def _sequence(rng, fps, seconds, events):
    """시각마다 관측 한 건. events = [(시작초, 끝초, 종류)], 겹쳐도 된다."""
    t, out = 0.0, []
    base = dict(ear=rng.uniform(0.27, 0.36), pitch=rng.uniform(-12, 18), yaw=rng.uniform(-10, 10))
    while t < seconds:
        o = {'face_detected': True, 'ear': base['ear'] + rng.gauss(0, 0.01), 'mar': 0.1 + abs(rng.gauss(0, 0.03)),
             'pitch': base['pitch'] + rng.gauss(0, 1.5), 'yaw': base['yaw'] + rng.gauss(0, 2.0),
             'detected_objects': []}
        for a, b, kind in events:
            if not a <= t < b:
                continue
            if kind == 'closed':
                o['ear'] = 0.08 + abs(rng.gauss(0, 0.01))
            elif kind == 'yawn':
                o['mar'] = 0.9 + rng.gauss(0, 0.05)
            elif kind == 'turn':
                o['yaw'] = base['yaw'] + 45 + rng.gauss(0, 3)
            elif kind == 'nod':
                o['pitch'] = base['pitch'] + 30 + rng.gauss(0, 2)
            elif kind == 'phone':
                o['detected_objects'] = [{'class': 'cell phone', 'confidence': 0.8, 'bbox': [0, 0, 1, 1]}]
            elif kind == 'noface':
                o = {'face_detected': False, 'ear': None, 'mar': None, 'pitch': None, 'yaw': None,
                     'detected_objects': []}
        out.append({'t': round(t, 6), 'obs': o})
        # 브라우저 프레임 간격은 고르지 않다. 가끔 길게 끊기는 것도 넣는다.
        t += (1.0 / fps) * rng.uniform(0.6, 1.6) + (rng.uniform(0.3, 1.2) if rng.random() < 0.01 else 0.0)
    return out


SEQUENCES = [
    ('30fps 깜빡임·2초 감김', 30, 40, [(5, 5.15, 'closed'), (9, 9.2, 'closed'), (14, 16.6, 'closed'), (30, 30.2, 'closed')]),
    ('15fps 하품·고개 돌림·숙임', 15, 40, [(6, 9, 'yawn'), (14, 18, 'turn'), (24, 28, 'nod'), (33, 36, 'yawn'), (34, 36, 'turn')]),
    ('8fps 누적 졸음(짧게 여러 번)', 8, 60, [(s, s + 1.7, 'closed') for s in range(6, 50, 2)]),
    ('24fps 얼굴 사라짐·휴대폰', 24, 40, [(1.0, 1.6, 'noface'), (10, 13, 'noface'), (20, 23, 'phone'), (30, 33.5, 'closed'), (31, 33, 'turn')]),
    ('60fps 보정 중 깜빡임', 60, 20, [(0.5, 0.7, 'closed'), (8, 10.5, 'closed'), (9, 10, 'yawn'), (9.5, 10.5, 'nod')]),
]

BLINK_SEQUENCES = [
    ('눈 감김 점수 30fps 웃음·2초 감김', 30, 40, [(5, 12, 'smile'), (8, 8.2, 'closed'), (20, 23, 'closed'), (30, 33, 'smile')]),
]


def main():
    rng = random.Random(SEED)

    geometry = []
    for _ in range(200):
        e, m = _points(rng, 6), _points(rng, 8)
        geometry.append({'eye': e, 'mouth': m, 'ear': D.ear(e), 'mar': D.mar(m)})

    angles = []
    for _ in range(200):
        m = _rotation(rng)
        pitch, yaw = D.head_angles(m)
        angles.append({'matrix': m, 'pitch': pitch, 'yaw': yaw})

    # 기존 벤치의 800건 + 연속 감김·얼굴 없음·경계값
    rows = [{'face_detected': True, 'closed_sec': 0.0, **{k: r[k] for k in ('mar', 'pitch', 'yaw', 'perclos', 'detected_objects')}}
            for r in make_dataset()]
    for _ in range(200):
        rows.append({'face_detected': rng.random() > 0.1, 'closed_sec': rng.choice([0.0, 0.5, 1.99, 2.0, 2.01, 5.0]),
                     'mar': rng.uniform(0, 1.2), 'pitch': rng.uniform(-40, 40), 'yaw': rng.uniform(-60, 60),
                     'perclos': rng.choice([0.0, 0.3, 0.6, 0.6000001, 0.9]),
                     'detected_objects': [{'class': 'book'}] if rng.random() < 0.15 else []})
    decisions = []
    for s in rows:
        flags = D.classify(s)
        level, reason = D.alert(flags, s['detected_objects'])
        decisions.append({'input': s, 'flags': flags, 'level': level, 'reason': reason})

    sequences = []
    for name, fps, seconds, events in SEQUENCES:
        frames = _sequence(rng, fps, seconds, events)
        mon = D.Monitor()
        sequences.append({'name': name, 'mode': 'ear', 'frames': frames,
                          'expected': [mon.update(f['t'], f['obs']) for f in frames]})

    # 눈 감김 점수 모드: 뜬 눈 0.9 안팎, 웃을 때 0.55(감김 아님), 감으면 0.1
    for name, fps, seconds, events in BLINK_SEQUENCES:
        frames = _sequence(rng, fps, seconds, [e for e in events if e[2] != 'smile'])
        for f in frames:
            o = f['obs']
            if not o['face_detected']:
                continue
            closed = o['ear'] < 0.15
            smiling = any(a <= f['t'] < b for a, b, k in events if k == 'smile')
            # 웃을 때 감김 점수가 0.7까지 올라도(작은 눈) 근육 점수 0.8이 같이 뜨면 감김으로 세지 않아야 한다.
            l = (0.9 if closed else 0.70 if smiling else 0.08) + rng.gauss(0, 0.02)
            sq = D.squint_score(0.5 if smiling else 0.05, 0.8 if smiling else 0.02, 0.7 if smiling else 0.03)
            o['ear'] = D.eye_openness(l, l + rng.gauss(0, 0.03), 0.0 if closed else sq)
        mon = D.Monitor(ear_ratio=D.BLINK_OPEN_RATIO, ear_range=D.BLINK_OPEN_RANGE, ear_thresh=D.BLINK_OPEN_THRESH)
        sequences.append({'name': name, 'mode': 'blink', 'frames': frames,
                          'expected': [mon.update(f['t'], f['obs']) for f in frames]})

    openness = []
    for _ in range(200):
        a = [rng.random() for _ in range(5)]
        sq = D.squint_score(a[2], a[3], a[4])
        openness.append({'args': a, 'squint': sq, 'open': D.eye_openness(a[0], a[1], sq)})

    out = os.path.join(ROOT, 'web', 'tests', 'fixture.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'geometry': geometry, 'angles': angles, 'openness': openness,
                   'decisions': decisions, 'sequences': sequences}, f, ensure_ascii=False)

    n_frames = sum(len(s['frames']) for s in sequences)
    print(f"기하 {len(geometry)} · 각도 {len(angles)} · 판단 {len(decisions)} · 시퀀스 {len(sequences)}개 {n_frames}프레임")
    for s in sequences:
        lv = [e['alert_level'] for e in s['expected']]
        print(f"  {s['name']}: 프레임 {len(lv)} · 레벨별 {[lv.count(k) for k in range(4)]}")
    print(f"저장: web/tests/fixture.json ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
