import pandas as pd
import numpy as np

DATA_PATH = "heart.csv"
RANDOM_STATE = 42

# Mapping dicts 
label_map = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "target": "Heart Disease"
}

sex_map = {0: "Female", 1: "Male", "F": "Female", "M": "Male"}
exang_map = {0: "No", 1: "Yes", "N": "No", "Y": "Yes"}
fbs_map = {0: "<= 120 mg/dl", 1: "> 120 mg/dl"}
target_map = {0: "No Disease", 1: "Disease"}
cp_map = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic",
    "TA": "Typical Angina",
    "ATA": "Atypical Angina",
    "NAP": "Non-anginal Pain",
    "ASY": "Asymptomatic",
}
restecg_map = {
    0: "Normal",
    1: "ST-T Abnormality",
    2: "LV Hypertrophy",
    "Normal": "Normal",
    "ST": "ST-T Abnormality",
    "LVH": "LV Hypertrophy",
}
slope_map = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping",
    "Up": "Upsloping",
    "Flat": "Flat",
    "Down": "Downsloping",
}
thal_map = {
    0: "Unknown",
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect",
}

CATEGORICAL_ENCODERS = {
    "sex": {"F": 0, "M": 1, 0: 0, 1: 1},
    "cp": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3, 0: 0, 1: 1, 2: 2, 3: 3},
    "fbs": {0: 0, 1: 1},
    "restecg": {"Normal": 0, "ST": 1, "LVH": 2, 0: 0, 1: 1, 2: 2},
    "exang": {"N": 0, "Y": 1, 0: 0, 1: 1},
    "slope": {"Up": 0, "Flat": 1, "Down": 2, 0: 0, 1: 1, 2: 2},
}

def load_data(path=DATA_PATH):
    """Load and preprocess the dataset."""
    df = pd.read_csv(path).drop_duplicates().copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize column names
    df = df.rename(columns={
        'chestpaintype': 'cp',
        'restingbp': 'trestbps',
        'cholesterol': 'chol',
        'fastingbs': 'fbs',
        'restingecg': 'restecg',
        'maxhr': 'thalach',
        'exerciseangina': 'exang',
        'oldpeak': 'oldpeak',
        'st_slope': 'slope',
        'heartdisease': 'target',
    })

    return df

def get_display_data():
    """Get dataframe with human-readable labels."""
    df = load_data()
    display_df = df.copy()

    display_df["sex_label"] = display_df["sex"].map(sex_map).fillna(display_df["sex"].astype(str))
    display_df["exang_label"] = display_df["exang"].map(exang_map).fillna(display_df["exang"].astype(str))
    display_df["target_label"] = display_df["target"].map(target_map).fillna(display_df["target"].astype(str))
    display_df["cp_label"] = display_df["cp"].map(cp_map).fillna(display_df["cp"].astype(str))
    display_df["restecg_label"] = display_df["restecg"].map(restecg_map).fillna(display_df["restecg"].astype(str))
    display_df["slope_label"] = display_df["slope"].map(slope_map).fillna(display_df["slope"].astype(str))
    if "thal" in display_df.columns:
        display_df["thal_label"] = display_df["thal"].map(thal_map).fillna(display_df["thal"].astype(str))

    return display_df

def get_random_state():
    """Return a fixed random state for reproducibility."""
    return RANDOM_STATE
