import cv2
from retinaface import RetinaFace
from numpy.dtypes import UInt8DType
from dotenv import load_dotenv
import os
from runtime.process import preprocess_image, postprocess_image
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__, log_file="logs/detector.log")

def detect_faces(img: UInt8DType, conf_thresh=float(os.getenv("CONFIDENCE_THRESHOLD", 0.9))):
    logger.info(f"Starting face detection with confidence threshold {conf_thresh}")
    if img is None:
        logger.error("Image not found or invalid path")
        raise ValueError("Image not found or invalid path")
    
    resolution = img.shape[:2]
    # img = preprocess_image(img)
    # logger.debug(f"Image preprocessed, resolution: {resolution}")

    # Run detector (threshold controls minimum confidence)
    detections = RetinaFace.detect_faces(img, threshold=conf_thresh)
    logger.info(f"Detector found {len(detections) if isinstance(detections, dict) else 0} potential faces")

    results = {
        "annotated_img": None,
        "detections": []
    }
    bboxes = []
    if isinstance(detections, dict):
        for face_id, face in detections.items():
            score = face.get("score", 0.0)
            logger.debug(f"Processing face {face_id} with score {score}")
            # Only accept detections above the confidence threshold
            if score < conf_thresh:
                logger.debug(f"Face {face_id} below threshold, skipping")
                continue

            # bounding box
            bbox = face.get("facial_area")
            
            # landmarks
            landmarks_dict = face.get("landmarks", {})
            landmarks = []
            for key in ['right_eye', 'left_eye', 'nose', 'mouth_right', 'mouth_left']:
                if key in landmarks_dict:
                    landmarks.extend(landmarks_dict[key])
            
            results["detections"].append({
                "bbox": bbox,
                "score": score,
                "landmarks": landmarks
            })
            bboxes.append(bbox)
            logger.debug(f"Added detection for face {face_id}")

    annotated_img = postprocess_image(img, bboxes, resolution)
    results["annotated_img"] = annotated_img
    logger.info(f"Face detection completed, {len(results['detections'])} detections above threshold")
    
    return results

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image_path = "test.png"
    img = cv2.imread(image_path)
    faces = detect_faces(img, conf_thresh=0.5)

    for detection in faces["detections"]:
        bbox = detection["bbox"]
        score = detection["score"]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()