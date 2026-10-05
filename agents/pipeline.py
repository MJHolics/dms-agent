"""
DMS Agent Pipeline — mediapipe Tasks API (0.10.x) 버전
face_landmarker.task 모델을 최초 1회 자동 다운로드합니다 (~30MB).
"""
import os
import time
import urllib.request
from typing import TypedDict, Optional, List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agents import decision

# ── 상수 ──────────────────────────────────────────
# 판단 규칙·임계·랜드마크 인덱스는 agents/decision.py가 원본이다(브라우저 이식본과 공유).
LEFT_EYE   = decision.LEFT_EYE
RIGHT_EYE  = decision.RIGHT_EYE
MOUTH      = decision.MOUTH
EAR_THRESH = decision.EAR_THRESH
MAR_THRESH = decision.MAR_THRESH
PERCLOS_THRESH = decision.PERCLOS_THRESH
DANGEROUS  = {67: 'cell phone', 73: 'book'}

# 프로젝트 루트 기준 모델 경로
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_ROOT, 'models', 'face_landmarker.task')
_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
               'face_landmarker/face_landmarker/float16/1/face_landmarker.task')


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
_ear = decision.ear
_mar = decision.mar


def _head_pose(matrix):
    """FaceLandmarker의 얼굴 변환 행렬 → (pitch, yaw). 숙임이 양수.

    예전에는 solvePnP 결과에 360을 곱해 실제 얼굴에서 수만 도가 나왔다
    (정면 사진에서 pitch -52628). 행렬에서 정면 벡터를 직접 읽도록 바꿨다.
    """
    return decision.head_angles(np.asarray(matrix).tolist())


def _check_ollama():
    try:
        urllib.request.urlopen('http://localhost:11434', timeout=1)
        return True
    except:
        return False


# ── 모델 싱글톤 ─────────────────────────────────
_face_landmarker = None
_yolo_model      = None


def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        if not os.path.exists(_MODEL_PATH):
            print('face_landmarker.task 다운로드 중 (~30MB)...')
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print('다운로드 완료')
        # 경로 대신 바이트로 넘긴다 — 한글이 든 절대 경로는 mediapipe 네이티브 쪽에서 못 연다.
        with open(_MODEL_PATH, 'rb') as f:
            base_options = mp_python.BaseOptions(model_asset_buffer=f.read())
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO('yolov8n.pt')
    return _yolo_model


# ── 에이전트 노드 ─────────────────────────────────
def face_analysis_agent(state: DMSState) -> DMSState:
    if state['frame'] is None:
        return {**state, 'face_detected': False}

    landmarker = _get_face_landmarker()
    rgb        = cv2.cvtColor(state['frame'], cv2.COLOR_BGR2RGB)
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result     = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {**state, 'face_detected': False, 'ear': None, 'mar': None}

    lm   = result.face_landmarks[0]          # List[NormalizedLandmark]
    h, w = state['frame'].shape[:2]

    le = [(lm[i].x*w, lm[i].y*h) for i in LEFT_EYE]
    re = [(lm[i].x*w, lm[i].y*h) for i in RIGHT_EYE]
    mo = [(lm[i].x*w, lm[i].y*h) for i in MOUTH]

    ear   = (_ear(le) + _ear(re)) / 2
    mar   = _mar(mo)
    mats = result.facial_transformation_matrixes
    pitch, yaw = _head_pose(mats[0]) if mats else (None, None)

    hist    = list(state.get('ear_history', [])) + [ear]
    hist    = hist[-900:]
    perclos = sum(1 for e in hist if e < EAR_THRESH) / len(hist)

    return {
        **state,
        'face_detected': True,
        'ear':     round(ear, 4),
        'mar':     round(mar, 4),
        'pitch':   round(pitch, 2) if pitch is not None else None,
        'yaw':     round(yaw, 2)   if yaw   is not None else None,
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
    return {**state, **decision.classify(state)}


def alert_manager_agent(state: DMSState) -> DMSState:
    lv, reason = decision.alert(state, state['detected_objects'])
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
