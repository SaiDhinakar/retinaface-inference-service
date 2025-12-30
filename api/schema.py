from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Face(BaseModel):
    bbox: BoundingBox
    score: float
    landmarks: List[float]

class FaceDetectionResponse(BaseModel):
    annotated_image: str
    faces: List[Face]
