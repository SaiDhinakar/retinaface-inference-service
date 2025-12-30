import cv2

def preprocess_image(img):
    """
    Resize image to have maximum dimension of 640 pixels.
    
    :param img: image array
    :return: resized image array
    """
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1:
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)))
    else:
        img_small = img
    
    return img_small

def postprocess_image(img, bboxes, resolution):
    """
    Draw bounding boxes on image and resize to original resolution.
    
    :param img: image array
    :param bboxes: list of bounding box coordinates [(x1, y1, x2, y2), ...]
    :param resolution: original image resolution (height, width)
    :return: annotated image array
    """

    annotated_img = img.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0))
    annotated_img = cv2.resize(annotated_img, (resolution[1], resolution[0]))
    return annotated_img