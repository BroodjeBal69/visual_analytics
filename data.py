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
    "ca": "Major Vessels",
    "thal": "Thalassemia",
    "target": "Heart Disease"
}

sex_map = {0: "Female", 1: "Male"}
exang_map = {0: "No", 1: "Yes"}
fbs_map = {0: "<= 120 mg/dl", 1: "> 120 mg/dl"}
target_map = {0: "No Disease", 1: "Disease"}
cp_map = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic",
}
restecg_map = {
    0: "Normal",
    1: "ST-T Abnormality",
    2: "LV Hypertrophy",
}
slope_map = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping",
}
thal_map = {
    0: "Unknown",
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect",
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

    # Add missing columns 
    for col in ['ca', 'thal']:
        if col not in df.columns:
            df[col] = np.nan

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
    display_df["thal_label"] = display_df["thal"].map(thal_map).fillna(display_df["thal"].astype(str))

    return display_df

def get_random_state():
    """Return a fixed random state for reproducibility."""
    return RANDOM_STATE