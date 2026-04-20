import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import DMSResult
from agents.pipeline import build_dms, initial_state

app = FastAPI(
    title="DMS Agent API",
    description="Driver Monitoring System — LangGraph 기반 실시간 운전자 상태 분석",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# 파이프라인 전역 로드 (서버 시작 시 1회)
_dms_app   = build_dms()
_dms_state = initial_state()


@app.get("/health")
def health():
    return {"status": "ok", "service": "DMS Agent"}


@app.post("/analyze", response_model=DMSResult)
async def analyze(file: UploadFile = File(...)):
    """
    이미지 프레임을 받아 운전자 상태 분석 결과 반환
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    global _dms_state
    t0 = time.perf_counter()
    _dms_state = _dms_app.invoke({
        **_dms_state,
        "frame":    frame,
        "frame_id": _dms_state.get("frame_id", 0) + 1
    })
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return DMSResult(
        frame_id=       _dms_state["frame_id"],
        face_detected=  _dms_state["face_detected"],
        ear=            _dms_state["ear"],
        mar=            _dms_state["mar"],
        yaw=            _dms_state["yaw"],
        pitch=          _dms_state["pitch"],
        perclos=        _dms_state["perclos"],
        detected_objects=_dms_state["detected_objects"],
        is_drowsy=      _dms_state["is_drowsy"],
        is_yawning=     _dms_state["is_yawning"],
        is_distracted=  _dms_state["is_distracted"],
        has_danger_obj= _dms_state["has_danger_obj"],
        alert_level=    _dms_state["alert_level"],
        alert_reason=   _dms_state["alert_reason"],
        llm_message=    _dms_state["llm_message"],
        inference_ms=   round(elapsed_ms, 2)
    )
