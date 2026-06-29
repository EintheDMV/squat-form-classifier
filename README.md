# Squat Pose Form Classifier

This project is a computer vision and pose-estimation prototype that analyzes an uploaded squat image and classifies the lifter’s form into one of six categories:

* Correct squat
* Shallow squat
* Forward lean
* Knees caving in
* Heels off ground
* Asymmetric squat

## Project Overview

The app uses MediaPipe Pose Landmarker to detect human body landmarks from an uploaded image. It then converts those landmarks into pose-based features such as knee angles, hip angles, ankle angles, torso lean, spine angle, symmetry score, knee lateral deviation, and hip depth.

Those extracted features are passed into a trained Random Forest classifier, which predicts the most likely squat form category and displays a confidence score, feedback, extracted pose features, and class probabilities.

## Tech Stack

* Python
* Streamlit
* MediaPipe Pose Landmarker
* scikit-learn
* pandas
* NumPy
* joblib
* PIL / Pillow

## Dataset

The model was trained using a squat exercise pose dataset in CSV format. The dataset contains pose-based features extracted from squat exercise frames, including joint angles, torso lean, spine angle, symmetry score, hip depth, frame metadata, and form labels.

Label mapping:

* 0 = Correct squat
* 1 = Shallow squat
* 2 = Forward lean
* 3 = Knees caving in
* 4 = Heels off ground
* 5 = Asymmetric squat

## How It Works

1. User uploads a squat image.
2. MediaPipe detects body landmarks.
3. The app extracts pose-based biomechanical features.
4. A trained Random Forest model classifies the squat form.
5. The app displays the prediction, confidence score, feedback, extracted features, and class probabilities.

## Deployment

The app is deployed using Streamlit Community Cloud and can be accessed through the hosted Streamlit link.

## Notes and Limitations

This project is a prototype for learning computer vision, pose estimation, and machine learning classification. It is not intended to replace a qualified coach, trainer, physical therapist, or medical professional.

Model performance may vary depending on image quality, camera angle, body visibility, clothing, lighting, and whether the uploaded image is similar to the training data.
