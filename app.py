import math

import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_FILE = "squat_pose_classifier.joblib"
POSE_MODEL_FILE = "pose_landmarker.task"


DEFAULT_LABEL_MAP = {
    0: "Correct squat",
    1: "Shallow squat",
    2: "Forward lean",
    3: "Knees caving in",
    4: "Heels off ground",
    5: "Asymmetric squat"
}


def angle_between_three_points(point_a, point_b, point_c):
    """
    Calculates the angle at point_b.

    Example:
    hip -> knee -> ankle gives the knee angle.
    """
    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return math.degrees(math.acos(cosine_angle))


def angle_from_vertical(top_point, bottom_point):
    """
    Calculates how far a body segment leans from vertical.

    0 degrees = vertical.
    Larger number = more lean.
    """
    top = np.array(top_point, dtype=float)
    bottom = np.array(bottom_point, dtype=float)

    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]

    return math.degrees(math.atan2(abs(dx), abs(dy)))


def point_to_line_distance(point, line_start, line_end):
    """
    Calculates how far a point is from a line.

    This is used as a rough knee-lateral-deviation feature.
    """
    p = np.array(point, dtype=float)
    a = np.array(line_start, dtype=float)
    b = np.array(line_end, dtype=float)

    line = b - a
    line_length = np.linalg.norm(line)

    if line_length == 0:
        return np.nan

    # 2D cross product magnitude
    cross_product = abs(line[0] * (a[1] - p[1]) - line[1] * (a[0] - p[0]))

    return cross_product / line_length


def landmark_to_pixel(landmark, image_width, image_height):
    """
    MediaPipe gives x and y values between 0 and 1.
    This converts them into actual pixel coordinates.
    """
    return np.array(
        [
            landmark.x * image_width,
            landmark.y * image_height
        ],
        dtype=float
    )


@st.cache_resource
def load_classifier():
    """
    Loads the trained squat classifier model.
    """
    bundle = joblib.load(MODEL_FILE)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        label_map = bundle.get("label_map", DEFAULT_LABEL_MAP)
        feature_medians = bundle.get("feature_medians", {})
        training_info = bundle.get("training_info", {})
    else:
        model = bundle
        label_map = DEFAULT_LABEL_MAP
        feature_medians = {}
        training_info = {}
        feature_columns = [
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
            "left_ankle_angle",
            "right_ankle_angle",
            "spine_angle",
            "torso_lean",
            "left_knee_lateral",
            "right_knee_lateral",
            "symmetry_score",
            "hip_depth"
        ]

    return model, feature_columns, label_map, feature_medians, training_info

@st.cache_resource
def create_pose_landmarker():
    """
    Creates the MediaPipe Pose Landmarker.
    """
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_FILE)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    return vision.PoseLandmarker.create_from_options(options)


def extract_squat_features_from_pil_image(pil_image):
    """
    Takes an uploaded image and converts it into squat pose features.
    """
    image_width, image_height = pil_image.size

    image_np = np.array(pil_image.convert("RGB"))

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    landmarker = create_pose_landmarker()
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise ValueError(
            "No human pose detected. Please upload a clear full-body squat image."
        )

    landmarks = result.pose_landmarks[0]

    # MediaPipe landmark index numbers
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    # Convert MediaPipe landmarks into pixel coordinates
    left_shoulder = landmark_to_pixel(landmarks[LEFT_SHOULDER], image_width, image_height)
    right_shoulder = landmark_to_pixel(landmarks[RIGHT_SHOULDER], image_width, image_height)

    left_hip = landmark_to_pixel(landmarks[LEFT_HIP], image_width, image_height)
    right_hip = landmark_to_pixel(landmarks[RIGHT_HIP], image_width, image_height)

    left_knee = landmark_to_pixel(landmarks[LEFT_KNEE], image_width, image_height)
    right_knee = landmark_to_pixel(landmarks[RIGHT_KNEE], image_width, image_height)

    left_ankle = landmark_to_pixel(landmarks[LEFT_ANKLE], image_width, image_height)
    right_ankle = landmark_to_pixel(landmarks[RIGHT_ANKLE], image_width, image_height)

    left_foot = landmark_to_pixel(landmarks[LEFT_FOOT_INDEX], image_width, image_height)
    right_foot = landmark_to_pixel(landmarks[RIGHT_FOOT_INDEX], image_width, image_height)

    # Midpoints
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2

    # Joint angles
    left_knee_angle = angle_between_three_points(left_hip, left_knee, left_ankle)
    right_knee_angle = angle_between_three_points(right_hip, right_knee, right_ankle)

    left_hip_angle = angle_between_three_points(left_shoulder, left_hip, left_knee)
    right_hip_angle = angle_between_three_points(right_shoulder, right_hip, right_knee)

    left_ankle_angle = angle_between_three_points(left_knee, left_ankle, left_foot)
    right_ankle_angle = angle_between_three_points(right_knee, right_ankle, right_foot)

    # Torso lean
    torso_lean = angle_from_vertical(mid_shoulder, mid_hip)

    # Spine angle approximation
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    spine_angle = abs(math.degrees(math.atan2(dy, dx)))

    # Knee lateral deviation
    left_leg_length = np.linalg.norm(left_hip - left_ankle)
    right_leg_length = np.linalg.norm(right_hip - right_ankle)

    left_knee_lateral = point_to_line_distance(left_knee, left_hip, left_ankle)
    right_knee_lateral = point_to_line_distance(right_knee, right_hip, right_ankle)

    if left_leg_length != 0:
        left_knee_lateral = left_knee_lateral / left_leg_length

    if right_leg_length != 0:
        right_knee_lateral = right_knee_lateral / right_leg_length

    # Symmetry score
    symmetry_score = (
        abs(left_knee_angle - right_knee_angle)
        + abs(left_hip_angle - right_hip_angle)
        + abs(left_ankle_angle - right_ankle_angle)
    )

    # Hip depth
    # In image coordinates, larger y means lower in the image.
    hip_depth = ((landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2)

    features = {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_ankle_angle": left_ankle_angle,
        "right_ankle_angle": right_ankle_angle,
        "spine_angle": spine_angle,
        "torso_lean": torso_lean,
        "left_knee_lateral": left_knee_lateral,
        "right_knee_lateral": right_knee_lateral,
        "symmetry_score": symmetry_score,
        "hip_depth": hip_depth,
    }

    features = {key: round(value, 4) for key, value in features.items()}

    return features


def get_feedback(predicted_label):
    """
    Converts the model's numeric prediction into beginner-friendly feedback.
    """
    feedback = {
        0: "The model classified this as a correct squat.",
        1: "The model detected a possible shallow squat. The hips may not be reaching enough depth.",
        2: "The model detected possible forward lean. The torso may be leaning forward more than expected.",
        3: "The model detected possible knees caving in.",
        4: "The model detected possible heels coming off the ground.",
        5: "The model detected possible asymmetry between the left and right sides."
    }

    return feedback.get(predicted_label, "No feedback available for this class.")


# -----------------------------
# Streamlit App
# -----------------------------

st.title("Squat Pose Form Classifier")

st.write(
    "Upload a clear full-body squat image. "
    "The app will estimate body pose, calculate squat features, "
    "and classify the squat form."
)

st.warning(
    "Prototype note: This is AI-assisted fitness feedback. "
    "It should not replace a qualified coach, trainer, physical therapist, "
    "or medical professional."
)

uploaded_file = st.file_uploader(
    "Upload a squat image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    try:
        model, feature_columns, label_map, feature_medians, training_info = load_classifier()

        features = extract_squat_features_from_pil_image(image)

        input_row = pd.DataFrame([features])

        # Force the feature columns to match the columns used during training
        input_row = input_row[feature_columns]

        # Fill any missing values using the medians saved during training
        for col in feature_columns:
            if input_row[col].isna().any():
                input_row[col] = input_row[col].fillna(feature_medians.get(col, 0))

        prediction = model.predict(input_row)[0]
        probabilities = model.predict_proba(input_row)[0]

        confidence = max(probabilities)

        st.subheader("Prediction")
        st.write(f"**Result:** {label_map[prediction]}")
        st.write(f"**Confidence:** {confidence:.2%}")

        st.subheader("Feedback")
        st.write(get_feedback(prediction))

        st.subheader("Extracted Pose Features")
        st.dataframe(input_row)

        st.subheader("Model Training Info")
        st.write(training_info)

        st.subheader("Class Probabilities")

        probability_table = pd.DataFrame({
            "Class": [label_map[class_id] for class_id in model.classes_],
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        st.dataframe(probability_table)

    except Exception as e:
        st.error(f"Could not process image: {e}")