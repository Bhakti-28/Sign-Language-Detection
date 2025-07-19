import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import cv2
from cvzone.HandTrackingModule import HandDetector

# Set Streamlit page layout
st.set_page_config(page_title="Sign Language Detection", layout="wide")

# Load models
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Error: Model '{model_path}' not found.")
        return None
    return tf.keras.models.load_model(model_path)

# Load ISL and ASL models
isl_model = load_model("MobileNetV2_ISL.keras")
asl_model = load_model("MobileNetV2_ASL.keras")

# Class names
isl_class_names = [str(i) for i in range(1, 11)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
asl_class_names = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + ["del", "space", "nothing"]

# Sidebar
st.sidebar.title("Select Feature")
page = st.sidebar.radio("Choose an option", [
    "Home", 
    "Static Detection", 
    "Live Detection"
])

# Home Page
if page == "Home":
    st.title("Welcome to the Sign Language Detection App")
    st.markdown("""
        This app detects hand gestures from **Indian Sign Language (ISL)** and **American Sign Language (ASL)**.

        ### Features:
        - Upload an image for ISL/ASL gesture detection.
        - Use webcam for real-time gesture recognition.

        ### How to Use:
        - Go to **Static Detection** to upload an image and classify the sign.
        - Go to **Live Detection** to use your webcam and detect signs in real time.

        > Built using TensorFlow, OpenCV, and Streamlit
    """)

# Unified Static Detection (ISL/ASL)
elif page == "Static Detection":
    st.title("Static Image Detection")

    language = st.selectbox("Select Language", ["ISL", "ASL"])
    uploaded_file = st.file_uploader(f"Upload a {language} image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = image.resize((299, 299))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if language == "ISL":
            model = isl_model
            class_names = isl_class_names
        else:
            model = asl_model
            class_names = asl_class_names

        if model:
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions)
            st.success(f"Predicted Sign: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2f}**")
        else:
            st.error(f"{language} model not loaded.")

# Unified Live Detection (ISL/ASL)
elif page == "Live Detection":
    st.title("Live Webcam Detection")

    language = st.selectbox("Select Language", ["ISL", "ASL"])
    run = st.checkbox("Start Webcam", value=False)

    if language == "ISL":
        model = isl_model
        class_names = isl_class_names
    else:
        model = asl_model
        class_names = asl_class_names

    frame_window = st.image([])
    prediction_area = st.empty()

    input_size = (299, 299)
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1, detectionCon=0.7)

    def preprocess_roi(roi):
        roi = cv2.resize(roi, input_size)
        roi = roi / 255.0
        return np.expand_dims(roi, axis=0)

    while run:
        success, frame = cap.read()
        if not success:
            st.warning("Webcam not accessible.")
            break

        frame = cv2.flip(frame, 1)
        hands, img = detector.findHands(frame, draw=True)

        prediction_text = ""
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            offset = 20
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(frame.shape[1], x + w + offset), min(frame.shape[0], y + h + offset)

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                preprocessed = preprocess_roi(roi)
                predictions = model.predict(preprocessed)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions)

                prediction_text = f"{predicted_class} ({confidence:.2f})"
                cv2.putText(img, prediction_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_window.image(img, channels="BGR")
        prediction_area.text(f"Prediction: {prediction_text}")

    cap.release()
