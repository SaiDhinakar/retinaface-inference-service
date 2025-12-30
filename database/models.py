from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func
from database.database import Base

class FaceRuntimeMetric(Base):
    __tablename__ = "face_runtime_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    inference_ms = Column(Float, nullable=False)
    response_ms = Column(Float, nullable=False)

    faces_detected = Column(Integer, nullable=False)

    backend = Column(String(16), nullable=False)