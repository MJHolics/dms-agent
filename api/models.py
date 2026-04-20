from pydantic import BaseModel
from typing import Optional, List


class DMSResult(BaseModel):
    frame_id:        int
    face_detected:   bool
    ear:             Optional[float]
    mar:             Optional[float]
    yaw:             Optional[float]
    pitch:           Optional[float]
    perclos:         Optional[float]
    detected_objects: List[dict]
    is_drowsy:       bool
    is_yawning:      bool
    is_distracted:   bool
    has_danger_obj:  bool
    alert_level:     int
    alert_reason:    str
    llm_message:     str
    inference_ms:    float
