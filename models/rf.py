import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
from data import load_data, get_random_state
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    from pygam import LogisticGAM, s
except Exception:
    LogisticGAM = None
    s = None

df = load_data()
RANDOM_STATE = get_random_state()

available_features = [col for col in df.columns if col != "target" and df[col].notna().any()]
X = df[available_features].copy()
y = df["target"]

categorical_features = [col for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"] if col in X.columns]
numeric_features = [c for c in X.columns if c not in categorical_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ]
)

def build_model_rf() -> Pipeline:
    """Return the Random Forest pipeline used across the app."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                random_state=RANDOM_STATE
            ))
        ]
    )

model_rf = build_model_rf()
model_rf.fit(X_train, y_train)
rf_pred = model_rf.predict(X_test)
rf_prob = model_rf.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

# Permutation importance on original features
perm = permutation_importance(
    model_rf, X_test, y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="roc_auc"
)
rf_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": perm.importances_mean
}).sort_values("importance", ascending=False)

gam_features = [feature for feature in ["age", "trestbps", "chol", "thalach", "oldpeak"] if feature in df.columns]
X_gam = df[gam_features].copy()
Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_gam, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


def build_model_gam():
    """Return a LogisticGAM model when pygam is available."""
    if LogisticGAM is None or s is None:
        return None

    terms = s(0)
    for idx in range(1, len(gam_features)):
        terms += s(idx)
    return LogisticGAM(terms)


model_gam = build_model_gam()
gam_accuracy = None
gam_auc = None

if model_gam is not None:
    model_gam.fit(Xg_train, yg_train)
    gam_pred = model_gam.predict(Xg_test)
    gam_prob = model_gam.predict_proba(Xg_test)
    gam_accuracy = accuracy_score(yg_test, gam_pred)
    gam_auc = roc_auc_score(yg_test, gam_prob)

__all__ = [
    "build_model_rf",
    "model_rf",
    "build_model_gam",
    "model_gam",
    "available_features",
    "categorical_features",
    "numeric_features",
    "gam_features",
    "rf_accuracy",
    "rf_auc",
    "gam_accuracy",
    "gam_auc",
    "rf_importance_df",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
]

def get_pca_features():
    """Get 2D PCA projection"""
    pca = PCA(n_components=2)
    return pca.fit_transform(X_scaled)

def get_patient_clusters():
    """Stratify patients into risk groups"""
    kmeans = KMeans(n_clusters=3)
    return kmeans.fit_predict(X_scaled)
