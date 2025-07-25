# Sign Language Detection App

A powerful and interactive Streamlit web application for recognizing hand gestures in both **Indian Sign Language (ISL)** and **American Sign Language (ASL)**. The app allows users to detect gestures from uploaded images or via real-time webcam input — making sign language detection intuitive and accessible.

## Model Research and Selection

To ensure high accuracy and optimal performance, we conducted extensive research and experimentation using a variety of state-of-the-art convolutional neural network (CNN) architectures:

- DenseNet169
- InceptionResNetV2
- InceptionV3
- ResNet50V2
- MobileNetV2
  
After evaluating each model on critical metrics — accuracy, precision, recall, and F1-score — across both the ISL and ASL datasets, **MobileNetV2** consistently outperformed the others.

## Features

- **Static Detection**  
  Upload an image of a hand gesture to detect the corresponding ISL/ASL sign.

- **Live Detection**  
  Use your webcam to detect hand signs in real-time using deep learning and computer vision.

- **Model Support**
  - ISL: `MobileNetV2_ISL.keras`  
  - ASL: `MobileNetV2_ASL.keras`

## Technologies Used

- Python
- Streamlit
- OpenCV
- Tensorflow / Keras
- CVZone


