
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


st.title('Object Detection Application')

upload_image = st.file_uploader('Please upload an Image....', type=['jpg', 'png', 'jpeg'])

MARGIN = 10  
ROW_SIZE = 10 
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0) 

def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

base_options = python.BaseOptions(model_asset_path="C:/Users/Elsayed Hassan/Downloads/efficientdet_lite0.tflite")
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

if upload_image is not None:
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image (numpy array) to PIL Image
    pil_img = Image.fromarray(img_rgb)

    # Show uploaded image
    st.image(pil_img, caption='Uploaded Image', use_container_width=True)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Perform object detection
    detection_result = detector.detect(mp_image)
    annotated_image = visualize(img_rgb.copy(), detection_result)

    # Convert annotated image back to PIL Image for Streamlit
    annotated_pil_img = Image.fromarray(annotated_image)

    # Show annotated image
    st.image(annotated_pil_img, caption='Annotated Image', use_container_width=True)
