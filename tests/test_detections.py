import requests
import cv2 as cv
import base64
import numpy as np

url = "http://localhost:8000/detect-from-img-bytes"

def test_detections_endpoint():
    img = cv.imread("test.png")
    if img is None:
        raise ValueError("Failed to load 'test.png'. Ensure the file exists and is a valid image.")
    
    _, img_encoded = cv.imencode('.jpg', img)  # Encode to JPEG bytes
    img_bytes = img_encoded.tobytes()  # Convert to bytes for transmission/decoding

    response = requests.post(url, data=img_bytes)
    

    assert response.status_code == 200

    data = response.json()
    assert "annotated_image" in data
    assert "faces" in data
    assert len(data["faces"]) > 0

    # save the resulting annotated image for manual inspection
    annotated_img_data = base64.b64decode(data["annotated_image"])
    with open("annotated_test_output_.jpg", "wb") as f:
        f.write(annotated_img_data)


if __name__ == "__main__":
    test_detections_endpoint()