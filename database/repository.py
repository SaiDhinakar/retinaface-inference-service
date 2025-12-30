from database.database import AsyncSessionLocal
from database.models import FaceRuntimeMetric

async def save_runtime_metric(
    *,
    inference_ms: float,
    response_ms: float,
    faces_detected: int,
    backend: str
):
    async with AsyncSessionLocal() as session:
        session.add(
            FaceRuntimeMetric(
                inference_ms=inference_ms,
                response_ms=response_ms,
                faces_detected=faces_detected,
                backend=backend
            )
        )
        await session.commit()