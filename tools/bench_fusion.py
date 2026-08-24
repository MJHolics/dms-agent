"""
bench_fusion.py — 복합신호 통합(Fusion) vs 단일 임계 결정 로직 실측 비교
=====================================================================

이 프로젝트의 핵심 주장은 "단순 규칙 임계(if-else)가 아니라 멀티에이전트가
복합 신호를 통합 판단한다"이다. 이 스크립트는 그 주장을 **수치로** 검증한다.

측정 대상은 지각(perception, MediaPipe/YOLO)이 아니라 **결정 계층**이다.
지각은 이미 검증돼 있고(EAR/MAR는 표준 공식), 이 프로젝트가 새로 주장하는 것은
"여러 신호를 어떻게 통합해 경보 레벨을 내는가"이기 때문이다. 그래서 평가 단위를
결정 로직으로 잡고, 실제 코드(`agents.pipeline`)의 판정 함수를 그대로 import해
라벨된 운전 상황 벡터에 태운다.

세 판정기 비교:
  B1  EAR 단일 임계        — 교과서적 졸음 감지 (EAR<0.25 → 경보)
  B2  EAR + PERCLOS        — 시간 평활 졸음 감지
  Fusion  실제 통합 로직   — state_classifier_agent + alert_manager_agent

정답(ground truth)은 융합 공식이 아니라 **안전 관점의 독립 스펙**으로 정의한다
(아래 SPEC). 따라서 Fusion도 100%가 아니며(단독 주의산만을 과소평가) 그 한계까지
정직하게 드러난다.

무료·로컬·재현. 외부 데이터·API·GPU 불필요. `python tools/bench_fusion.py`.
"""
import os
import sys
import json
import time
import random

# 결정 계층은 모델을 호출하지 않으므로 무거운 모델 로드 없이 실제 코드를 그대로 쓴다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.pipeline import state_classifier_agent, alert_manager_agent  # noqa: E402

SEED = 42
N_PER_CATEGORY = 100          # 카테고리당 표본 수
EAR_THRESH = 0.25             # pipeline.py와 동일
PERCLOS_THRESH = 0.15

# ── 안전 스펙(독립 정의) ──────────────────────────────────────────────
# 각 상황이 어떤 신호 분포를 갖고, 안전상 어떤 경보 레벨이 "옳은지"를 정의한다.
# gt(정답)는 사람 안전 담당자가 매길 값이며 Fusion 공식과 무관하게 정한다.
#   L0 정상 · L1 주의 · L2 위험 · L3 즉시정지
# jitter는 seeded RNG로 카테고리 내 자연스러운 변동을 준다.
def _rng_signal(rng, *, ear, mar, pitch, yaw, perclos, phone):
    """카테고리 중심값 주변에 현실적 변동을 준 신호 벡터 하나를 만든다."""
    def j(v, s):  # 가우시안 지터
        return round(v + rng.gauss(0, s), 4)
    objs = []
    if phone:
        objs = [{'class': 'cell phone', 'confidence': round(rng.uniform(0.5, 0.9), 3),
                 'bbox': [0, 0, 10, 10]}]
    return {
        'ear':     max(0.05, j(ear, 0.015)),
        'mar':     max(0.05, j(mar, 0.03)),
        'pitch':   j(pitch, 4.0),
        'yaw':     j(yaw, 6.0),
        'perclos': min(1.0, max(0.0, j(perclos, 0.02))),
        'detected_objects': objs,
    }

# (카테고리명, 정답레벨, 신호 생성 파라미터, 설명)
SPEC = [
    ('normal',       0, dict(ear=0.32, mar=0.35, pitch=2,  yaw=3,  perclos=0.02, phone=False),
     '정상 주행 — 눈 뜸·정면·객체 없음'),
    ('brief_blink',  0, dict(ear=0.20, mar=0.35, pitch=2,  yaw=3,  perclos=0.03, phone=False),
     '순간 깜빡임 — EAR는 낮지만 PERCLOS 정상(졸음 아님)'),
    ('yawn',         1, dict(ear=0.30, mar=0.72, pitch=3,  yaw=4,  perclos=0.04, phone=False),
     '하품 — 초기 피로 신호'),
    ('drowsy',       3, dict(ear=0.18, mar=0.40, pitch=6,  yaw=4,  perclos=0.28, phone=False),
     '지속 졸음 — 눈 감김 지속·PERCLOS 높음'),
    ('distracted',   2, dict(ear=0.31, mar=0.36, pitch=6,  yaw=42, perclos=0.03, phone=False),
     '주의산만 — 고개 이탈, 눈은 뜸(EAR 무력)'),
    ('phone',        3, dict(ear=0.30, mar=0.36, pitch=8,  yaw=6,  perclos=0.03, phone=True),
     '휴대폰 사용 — 눈 뜸·정면, 위험 객체(EAR 무력)'),
    ('phone_down',   3, dict(ear=0.29, mar=0.37, pitch=26, yaw=8,  perclos=0.03, phone=True),
     '고개 숙여 폰 조작 — 객체+고개숙임'),
    ('drowsy_distr', 3, dict(ear=0.17, mar=0.42, pitch=7,  yaw=40, perclos=0.30, phone=False),
     '졸음+주의산만 복합 이상'),
]


def make_dataset():
    rng = random.Random(SEED)
    rows = []
    for name, gt, params, _desc in SPEC:
        for _ in range(N_PER_CATEGORY):
            sig = _rng_signal(rng, **params)
            rows.append({'category': name, 'gt': gt, **sig})
    return rows


# ── 판정기 ────────────────────────────────────────────────────────────
def decide_ear_only(r):
    """B1: 교과서적 단일 임계 졸음 감지. EAR<0.25면 경보(L3), 아니면 정상.
    MAR·고개·객체·PERCLOS를 보지 않는다."""
    return 3 if r['ear'] < EAR_THRESH else 0


def decide_ear_perclos(r):
    """B2: 시간 평활 졸음 감지. PERCLOS로 순간 깜빡임 오경보를 억제하되
    여전히 주의산만·객체는 못 본다."""
    if r['perclos'] > PERCLOS_THRESH:
        return 3
    if r['ear'] < EAR_THRESH:
        return 1
    return 0


def _to_state(r):
    return {
        'face_detected': True,
        'ear': r['ear'], 'mar': r['mar'], 'pitch': r['pitch'], 'yaw': r['yaw'],
        'perclos': r['perclos'], 'detected_objects': r['detected_objects'],
    }


def decide_fusion(r):
    """Fusion(교체 후): 실제 코드(state_classifier_agent → alert_manager_agent)를 그대로 태운다."""
    state = state_classifier_agent(_to_state(r))
    state = alert_manager_agent(state)
    return state['alert_level']


def decide_fusion_legacy(r):
    """Fusion(교체 전): is_drowsy를 순간 EAR 단발 하강까지 잡던 옛 로직 재현.
    벤치마크로 이 버전의 순간 깜빡임 오경보를 발견해 실제 코드를 교체했다.
    분류만 옛 조건으로 대체하고 alert_manager는 실제 코드를 재사용한다."""
    st = _to_state(r)
    d = (st['ear'] or 1) < EAR_THRESH or (st['perclos'] or 0) > PERCLOS_THRESH  # 교체 전 조건
    y = (st['mar'] or 0) > MAR_THRESH
    i = abs(st['yaw'] or 0) > 30 or (st['pitch'] or 0) > 20
    o = len(st['detected_objects']) > 0
    st = {**st, 'is_drowsy': d, 'is_yawning': y, 'is_distracted': i,
          'has_danger_obj': o, 'risk_count': sum([d, y, i, o])}
    return alert_manager_agent(st)['alert_level']


MAR_THRESH = 0.60  # legacy 재현용 (pipeline.py와 동일)

DECIDERS = [
    ('B1_EAR단일', decide_ear_only),
    ('B2_EAR+PERCLOS', decide_ear_perclos),
    ('Fusion_교체전', decide_fusion_legacy),
    ('Fusion_교체후', decide_fusion),
]


# ── 평가 ──────────────────────────────────────────────────────────────
def evaluate(rows, decider):
    n = len(rows)
    exact = 0
    danger_total = 0        # gt >= 2
    danger_missed = 0       # gt >= 2 & pred < 2 (위험 미escalation)
    danger_silent = 0       # gt >= 2 & pred == 0 (완전 무경보 — 치명적)
    normal_total = 0        # gt == 0
    false_alarm = 0         # gt == 0 & pred >= 1
    per_cat = {}
    t0 = time.perf_counter()
    for r in rows:
        pred = decider(r)
        gt = r['gt']
        if pred == gt:
            exact += 1
        if gt >= 2:
            danger_total += 1
            if pred < 2:
                danger_missed += 1
            if pred == 0:
                danger_silent += 1
        if gt == 0:
            normal_total += 1
            if pred >= 1:
                false_alarm += 1
        c = per_cat.setdefault(r['category'], {'n': 0, 'exact': 0})
        c['n'] += 1
        c['exact'] += int(pred == gt)
    elapsed = time.perf_counter() - t0
    return {
        'exact_acc': round(exact / n, 4),
        'missed_danger_rate': round(danger_missed / danger_total, 4),
        'silent_danger_rate': round(danger_silent / danger_total, 4),
        'false_alarm_rate': round(false_alarm / normal_total, 4),
        'us_per_decision': round(elapsed / n * 1e6, 2),
        'per_category_acc': {k: round(v['exact'] / v['n'], 3) for k, v in per_cat.items()},
    }


def main():
    rows = make_dataset()
    results = {name: evaluate(rows, fn) for name, fn in DECIDERS}

    # ── 콘솔 표 ──
    print(f"\n데이터셋: {len(SPEC)}개 상황 × {N_PER_CATEGORY} = {len(rows)}개 (seed={SEED})\n")
    hdr = f"{'판정기':<18}{'정확도':>8}{'위험미탐':>10}{'무경보':>9}{'오경보':>9}{'us/건':>9}"
    print(hdr)
    print('-' * len(hdr))
    for name, m in results.items():
        print(f"{name:<18}{m['exact_acc']*100:>7.1f}%{m['missed_danger_rate']*100:>9.1f}%"
              f"{m['silent_danger_rate']*100:>8.1f}%{m['false_alarm_rate']*100:>8.1f}%"
              f"{m['us_per_decision']:>9.2f}")
    print("\n위험미탐 = 위험상황(정답≥위험)을 위험으로 못 올린 비율")
    print("무경보   = 위험상황인데 경보를 아예 안 낸 비율(치명적)")
    print("오경보   = 정상인데 경보를 낸 비율(경보 피로)\n")

    # 카테고리별
    print("카테고리별 정확도")
    cats = [c[0] for c in SPEC]
    print(f"{'상황':<14}" + "".join(f"{n:>16}" for n, _ in DECIDERS))
    for cat in cats:
        line = f"{cat:<14}"
        for name, _ in DECIDERS:
            line += f"{results[name]['per_category_acc'][cat]*100:>15.0f}%"
        print(line)

    # ── 저장 ──
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'reports')
    os.makedirs(outdir, exist_ok=True)
    payload = {
        'seed': SEED, 'n_per_category': N_PER_CATEGORY, 'n_total': len(rows),
        'spec': [{'category': n, 'gt': g, 'desc': d} for n, g, _p, d in SPEC],
        'results': results,
    }
    with open(os.path.join(outdir, 'fusion_bench.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: reports/fusion_bench.json")

    _plot(results, outdir)


def _plot(results, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        matplotlib.rcParams['axes.unicode_minus'] = False
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib 없음, 그래프 생략: {e})")
        return

    names = list(results.keys())
    metrics = [
        ('exact_acc', '정확도(↑)'),
        ('missed_danger_rate', '위험 미탐(↓)'),
        ('silent_danger_rate', '무경보(↓)'),
        ('false_alarm_rate', '오경보(↓)'),
    ]
    import numpy as np
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.2
    colors = ['#2e7d32', '#c62828', '#8e24aa', '#f9a825']
    for i, (key, label) in enumerate(metrics):
        vals = [results[n][key] * 100 for n in names]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('%')
    ax.set_ylim(0, 105)
    ax.set_title('결정 로직 비교 — 단일 임계 vs 복합신호 통합(Fusion)')
    ax.legend(ncol=4, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    path = os.path.join(outdir, 'fusion_bench.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    print(f"그래프 저장: reports/fusion_bench.png")


if __name__ == '__main__':
    main()
