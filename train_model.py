import os
import joblib
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# Change this if your CSV file has a different name
# --------------------------------------------------

CSV_FILE = "squat_features_augmented.csv"
MODEL_OUTPUT_FILE = "squat_pose_classifier.joblib"


# --------------------------------------------------
# Label names from the Kaggle README
# --------------------------------------------------

label_map = {
    0: "Correct squat",
    1: "Shallow squat",
    2: "Forward lean",
    3: "Knees caving in",
    4: "Heels off ground",
    5: "Asymmetric squat"
}


# --------------------------------------------------
# Features your Streamlit app creates
# --------------------------------------------------
# These column names must match the features created in app.py.

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


print("Current folder:")
print(os.getcwd())

print("\nFiles in this folder:")
print(os.listdir())


# --------------------------------------------------
# Load CSV
# --------------------------------------------------

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Could not find {CSV_FILE}. "
        "Make sure your Kaggle CSV is in this same folder, "
        "or change CSV_FILE at the top of train_model.py."
    )

df = pd.read_csv(CSV_FILE)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# --------------------------------------------------
# Check required columns
# --------------------------------------------------

if "label" not in df.columns:
    raise ValueError("The CSV must contain a column named 'label'.")

missing_features = [col for col in feature_columns if col not in df.columns]

if missing_features:
    raise ValueError(
        "Your CSV is missing these required feature columns:\n"
        + str(missing_features)
    )


# --------------------------------------------------
# Prepare X and y
# --------------------------------------------------

X = df[feature_columns].copy()
y = df["label"].copy()

# Convert all feature columns to numbers.
# If a value cannot be converted, it becomes NaN.
X = X.apply(pd.to_numeric, errors="coerce")

print("\nLabel counts:")
print(y.value_counts().sort_index())

print("\nMissing values before fill:")
print(X.isna().sum())


# --------------------------------------------------
# Fill missing values BEFORE training
# --------------------------------------------------
# This replaces the need for SimpleImputer.
# We save the median values so the app can use the same medians later if needed.

feature_medians = X.median(numeric_only=True)
X = X.fillna(feature_medians)

print("\nMissing values after fill:")
print(X.isna().sum())


# --------------------------------------------------
# Split train/test
# --------------------------------------------------

if "video_file" in df.columns:
    groups = df["video_file"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("\nUsed group split by video_file.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nUsed regular stratified split.")

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# --------------------------------------------------
# Train model
# --------------------------------------------------
# No SimpleImputer. Only RandomForestClassifier.

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)
print("Model training complete.")


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

y_pred = model.predict(X_test)

target_names = [label_map[i] for i in sorted(label_map.keys())]

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=sorted(label_map.keys()),
    target_names=target_names,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=sorted(label_map.keys())
))


# --------------------------------------------------
# Save local model bundle
# --------------------------------------------------

model_bundle = {
    "model": model,
    "feature_columns": feature_columns,
    "label_map": label_map,
    "feature_medians": feature_medians.to_dict(),
    "training_info": {
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "csv_file": CSV_FILE,
        "model_type": "RandomForestClassifier without SimpleImputer"
    }
}

joblib.dump(model_bundle, MODEL_OUTPUT_FILE)

print(f"\nSaved new local model as: {MODEL_OUTPUT_FILE}")
print("\nTraining info:")
print(model_bundle["training_info"])

print("\nImportant:")
print("This model does NOT contain SimpleImputer.")
print("\nNext run:")
print("python -m streamlit run app.py")