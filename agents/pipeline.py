"""
DMS Agent Pipeline
Notebook 04의 코드를 모듈화한 버전 (FastAPI 서빙용)
"""
import time
from typing import TypedDict, Optional, List

import cv2
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
from ultralytics import YOLO
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ── 상수 ──────────────────────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33, 160, 158, 133, 153, 144]
MOUTH      = [61, 291, 13, 14, 17, 0, 402, 178]
EAR_THRESH = 0.25
MAR_THRESH = 0.60
DANGEROUS  = {67: 'cell phone', 73: 'book'}


# ── State ─────────────────────────────────────────
class DMSState(TypedDict):
    frame:            Optional[np.ndarray]
    frame_id:         int
    face_detected:    bool
    ear:              Optional[float]
    mar:              Optional[float]
    pitch:            Optional[float]
    yaw:              Optional[float]
    perclos:          Optional[float]
    detected_objects: List[dict]
    is_drowsy:        bool
    is_yawning:       bool
    is_distracted:    bool
    has_danger_obj:   bool
    risk_count:       int
    alert_level:      int
    alert_reason:     str
    llm_message:      str
    ear_history:      List[float]


def initial_state(frame=None, frame_id=0) -> DMSState:
    return DMSState(
        frame=frame, frame_id=frame_id,
        face_detected=False, ear=None, mar=None,
        pitch=None, yaw=None, perclos=None,
        detected_objects=[], is_drowsy=False, is_yawning=False,
        is_distracted=False, has_danger_obj=False, risk_count=0,
        alert_level=0, alert_reason='정상', llm_message='', ear_history=[]
    )


# ── 헬퍼 ──────────────────────────────────────────
def _ear(p):
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A + B) / (2 * C)


def _mar(p):
    A = dist.euclidean(p[1], p[7])
    B = dist.euclidean(p[2], p[6])
    C = dist.euclidean(p[3], p[5])
    D = dist.euclidean(p[0], p[4])
    return (A + B + C) / (2 * D)


def _head_pose(lm, shape):
    h, w = shape[:2]
    m = np.array([[0,0,0],[0,-330,-65],[-225,170,-135],
                  [225,170,-135],[-150,-150,-125],[150,-150,-125]], dtype=np.float64)
    p = np.array([[lm[i].x*w, lm[i].y*h] for i in [1,152,33,263,61,291]], dtype=np.float64)
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    ok, rv, _ = cv2.solvePnP(m, p, cam, np.zeros((4,1)))
    if not ok:
        return None, None, None
    rm, _ = cv2.Rodrigues(rv)
    a, *_ = cv2.RQDecomp3x3(rm)
    return a[0]*360, a[1]*360, a[2]*360


def _check_ollama():
    try:
        import urllib.request
        urllib.request.urlopen('http://localhost:11434', timeout=1)
        return True
    except:
        return False


# ── 모델 싱글톤 ───────────────────────────────────
_face_mesh = None
_yolo_model = None


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    return _face_mesh


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO('yolov8n.pt')
    return _yolo_model


# ── 에이전트 노드 ─────────────────────────────────
def face_analysis_agent(state: DMSState) -> DMSState:
    if state['frame'] is None:
        return {**state, 'face_detected': False}

    mesh = _get_face_mesh()
    res  = mesh.process(cv2.cvtColor(state['frame'], cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        return {**state, 'face_detected': False, 'ear': None, 'mar': None}

    lm   = res.multi_face_landmarks[0].landmark
    h, w = state['frame'].shape[:2]

    le = [(lm[i].x*w, lm[i].y*h) for i in LEFT_EYE]
    re = [(lm[i].x*w, lm[i].y*h) for i in RIGHT_EYE]
    mo = [(lm[i].x*w, lm[i].y*h) for i in MOUTH]

    ear   = (_ear(le) + _ear(re)) / 2
    mar   = _mar(mo)
    pitch, yaw, _ = _head_pose(lm, state['frame'].shape)

    hist    = list(state.get('ear_history', [])) + [ear]
    hist    = hist[-900:]
    perclos = sum(1 for e in hist if e < EAR_THRESH) / len(hist)

    return {
        **state,
        'face_detected': True,
        'ear':     round(ear, 4),
        'mar':     round(mar, 4),
        'pitch':   round(pitch, 2) if pitch else None,
        'yaw':     round(yaw, 2)   if yaw   else None,
        'perclos': round(perclos, 4),
        'ear_history': hist
    }


def object_detection_agent(state: DMSState) -> DMSState:
    if state['frame'] is None or state['frame_id'] % 3 != 0:
        return {**state, 'detected_objects': state.get('detected_objects', [])}

    yolo  = _get_yolo()
    res   = yolo(state['frame'], verbose=False)[0]
    objs  = []
    for box in res.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        if cid in DANGEROUS and conf >= 0.45:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            objs.append({'class': DANGEROUS[cid], 'confidence': round(conf, 3),
                         'bbox': [x1, y1, x2, y2]})
    return {**state, 'detected_objects': objs}


def state_classifier_agent(state: DMSState) -> DMSState:
    if not state['face_detected']:
        return {**state, 'is_drowsy': False, 'is_yawning': False,
                'is_distracted': False, 'has_danger_obj': False, 'risk_count': 1}

    d = (state['ear'] or 1) < EAR_THRESH or (state['perclos'] or 0) > 0.15
    y = (state['mar'] or 0) > MAR_THRESH
    i = abs(state['yaw'] or 0) > 30 or (state['pitch'] or 0) > 20
    o = len(state['detected_objects']) > 0

    return {**state, 'is_drowsy': d, 'is_yawning': y,
            'is_distracted': i, 'has_danger_obj': o,
            'risk_count': sum([d, y, i, o])}


def alert_manager_agent(state: DMSState) -> DMSState:
    r    = state['risk_count']
    perc = state['perclos'] or 0
    obj  = state['detected_objects']

    if state['has_danger_obj']:   lv = 3; reason = f"위험 물체: {obj[0]['class']}"
    elif perc > 0.15 or r >= 3:  lv = 3; reason = '심각한 졸음 운전'
    elif r == 2:                  lv = 2; reason = '복합 위험 신호'
    elif r == 1:                  lv = 1; reason = '주의 필요'
    else:                         lv = 0; reason = '정상'

    return {**state, 'alert_level': lv, 'alert_reason': reason}


def llm_reasoning_agent(state: DMSState) -> DMSState:
    if state['alert_level'] == 0:
        return {**state, 'llm_message': '정상 운전 중입니다.'}

    msgs = {1: '주의가 필요합니다. 집중력을 유지하세요.',
            2: '위험 신호 감지! 잠시 휴식을 권장합니다.',
            3: '즉시 안전한 곳에 정차하세요! 매우 위험합니다!'}

    if not _check_ollama():
        return {**state, 'llm_message': msgs[state['alert_level']]}

    try:
        llm     = ChatOllama(model='qwen2.5:7b', temperature=0.3)
        reasons = []
        if state['is_drowsy']:     reasons.append('졸음')
        if state['is_yawning']:    reasons.append('하품')
        if state['is_distracted']: reasons.append('전방이탈')
        if state['has_danger_obj']:reasons.append('위험물체')
        prompt = (f"운전자 모니터링 시스템입니다. "
                  f"경고레벨:{state['alert_level']}/3, 감지:{','.join(reasons)}. "
                  f"한 문장으로 경고 메시지 생성.")
        res = llm.invoke([HumanMessage(content=prompt)])
        return {**state, 'llm_message': res.content}
    except:
        return {**state, 'llm_message': msgs[state['alert_level']]}


# ── 파이프라인 빌드 ────────────────────────────────
def build_dms():
    g = StateGraph(DMSState)
    g.add_node('face_analysis',    face_analysis_agent)
    g.add_node('object_detection', object_detection_agent)
    g.add_node('state_classifier', state_classifier_agent)
    g.add_node('alert_manager',    alert_manager_agent)
    g.add_node('llm_reasoning',    llm_reasoning_agent)
    g.set_entry_point('face_analysis')
    g.add_edge('face_analysis',    'object_detection')
    g.add_edge('object_detection', 'state_classifier')
    g.add_edge('state_classifier', 'alert_manager')
    g.add_edge('alert_manager',    'llm_reasoning')
    g.add_edge('llm_reasoning',    END)
    return g.compile()
