import cv2
import numpy as np
from retinaface import RetinaFace
from numpy.dtypes import UInt8DType
from dotenv import load_dotenv
import os
# uncomment if you want postprocessing of annotated images
# from runtime.process import postprocess_image
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__, log_file="logs/detector.log")

def align_face(img, landmarks, output_size=(112, 112)):
    """
    Align face using facial landmarks (eye positions).
    
    Args:
        img: Input image
        landmarks: List of landmark coordinates [right_eye_x, right_eye_y, left_eye_x, left_eye_y, ...]
        output_size: Desired output size (width, height)
    
    Returns:
        Aligned face image
    """
    if len(landmarks) < 4:
        logger.warning("Insufficient landmarks for alignment, returning None")
        return None
    
    # Extract eye coordinates (right_eye and left_eye are first 4 values)
    right_eye = np.array(landmarks[0:2])
    left_eye = np.array(landmarks[2:4])
    
    # Calculate angle between eyes
    dY = left_eye[1] - right_eye[1]
    dX = left_eye[0] - right_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Calculate the center point between the eyes
    eyes_center = ((right_eye[0] + left_eye[0]) // 2,
                   (right_eye[1] + left_eye[1]) // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    
    # Perform affine transformation
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    
    # Calculate the distance between eyes for scaling
    eye_distance = np.linalg.norm(right_eye - left_eye)
    
    # Desired eye distance in output (roughly 1/3 of output width)
    desired_eye_distance = output_size[0] * 0.35
    scale = desired_eye_distance / eye_distance
    
    # Apply rotation to eye positions to get new center
    eyes_center_rotated = np.dot(M[:, :2], np.array([eyes_center[0], eyes_center[1]])) + M[:, 2]
    
    # Calculate crop region centered on eyes
    w, h = output_size
    x = int(eyes_center_rotated[0] - w / 2)
    y = int(eyes_center_rotated[1] - h / 2.5)  # Eyes typically in upper portion
    
    # Ensure crop region is within bounds
    x = max(0, min(x, aligned.shape[1] - w))
    y = max(0, min(y, aligned.shape[0] - h))
    
    # Crop and resize
    cropped = aligned[y:y+h, x:x+w]
    
    # Resize to output size if needed
    if cropped.shape[:2] != output_size[::-1]:
        cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)
    
    logger.debug(f"Face aligned with angle {angle:.2f} degrees")
    return cropped

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
        "annotated_img": img.copy(),
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
            
            # Align the face using landmarks
            aligned_face = None
            if len(landmarks) >= 4:  # Need at least eye landmarks
                aligned_face = align_face(img, landmarks)
            
            results["detections"].append({
                "bbox": bbox,
                "score": score,
                "landmarks": landmarks,
                "aligned_face": aligned_face
            })
            bboxes.append(bbox)
            logger.debug(f"Added detection for face {face_id} with alignment")

    # Uncomment if annotated image output is needed
    # annotated_img = postprocess_image(img, bboxes, resolution)
    # results["annotated_img"] = annotated_img
    logger.info(f"Face detection completed, {len(results['detections'])} detections above threshold")
    
    return results

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image_path = "/mnt/sda1/retinaface-inference-service/tests/test.png"
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