from warnings import filterwarnings
filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import cv2
import numpy as np
import base64
import time

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from runtime.detector import detect_faces
from utils.device import get_device
from database import init_db
from database.repository import save_runtime_metric
from api.schema import Face, BoundingBox, FaceDetectionResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db.create_tables()
    yield

app = FastAPI(
    title="Face Detection Service",
    version="1.0.0",
    lifespan=lifespan
)

# -------- HEALTH CHECK ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- FACE DETECTION ----------
@app.post("/detect-from-image", response_model=FaceDetectionResponse)
async def detect_faces_from_image(file: UploadFile = File(...)):
    start_total = time.perf_counter()

    data = await file.read()

    img = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    # inference
    t0 = time.perf_counter()
    detections = detect_faces(img)
    inf_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - start_total) * 1000

    faces_out = []
    for det in detections["detections"]:
        bbox = det["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < 20 or h < 20:
            continue
        faces_out.append({
            "bbox": {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
            },
            "score": float(det["score"]),
            "landmarks": [float(x) for x in det["landmarks"]]
        })

    await save_runtime_metric(
        inference_ms=inf_ms,
        response_ms=total_ms,
        faces_detected=len(faces_out),
        backend=get_device()
    )

    res = {
        "annotated_image": base64.b64encode(cv2.imencode('.jpg', detections["annotated_img"])[1]).decode('utf-8'),
        "faces": faces_out
    }

    return res


@app.post("/detect-from-img-bytes", response_model=FaceDetectionResponse)
async def detect_faces_from_image(request: Request):
    start_total = time.perf_counter()

    img_bytes = await request.body()

    img_decoded = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img_decoded is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # inference
    t0 = time.perf_counter()
    detections = detect_faces(img_decoded)
    inf_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - start_total) * 1000

    faces_out = []
    for det in detections["detections"]:
        bbox = det["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < 20 or h < 20:
            continue
        faces_out.append({
            "bbox": {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
            },
            "score": float(det["score"]),
            "landmarks": [float(x) for x in det["landmarks"]]
        })

    await save_runtime_metric(
        inference_ms=inf_ms,
        response_ms=total_ms,
        faces_detected=len(faces_out),
        backend=get_device()
    )

    return {
        "annotated_image": None if detections["annotated_img"] is None else base64.b64encode(
            cv2.imencode(".jpg", detections["annotated_img"])[1]
        ).decode(),
        "faces": faces_out
    }

