import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image

# Initialize Roboflow with your API key
rf = Roboflow(api_key="ZYHlUFerIyHiH5T6qfOh")  # Replace with your actual API key
project = rf.workspace().project("blood-cell-detection-1ekwu-phux7")  # Use your project ID
model = project.version(1).model  # Use the correct version number

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_image(image):
    # Convert image to a numpy array
    img_array = np.array(image)

    # Perform inference using the model
    predictions = model.predict(img_array, confidence=40, overlap=30).json()
    return predictions

def draw_boxes(image, predictions):
    for pred in predictions['predictions']:
        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        width = int(pred['width'])
        height = int(pred['height'])
        class_name = pred['class']
        confidence = pred['confidence']

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.putText(image, f"{class_name}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Streamlit UI
st.title("Object Detection with Roboflow")
st.write("Upload an image to see the object detection results.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        predictions = predict_image(image)
        result_image = draw_boxes(np.array(image.copy()), predictions)

        st.image(result_image, caption='Predicted Result', use_column_width=True)
