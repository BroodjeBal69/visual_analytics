import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
from data import CATEGORICAL_ENCODERS, get_random_state, load_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    from pygam import LogisticGAM, f, s
except Exception:
    LogisticGAM = None
    f = None
    s = None

df = load_data()
RANDOM_STATE = get_random_state()


def _safe_mode(series: pd.Series, fallback: int = 0) -> int:
    cleaned = series.dropna()
    if cleaned.empty:
        return fallback
    modes = cleaned.mode()
    if modes.empty:
        return fallback
    return int(modes.iloc[0])

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

gam_numeric_features = [feature for feature in ["age", "trestbps", "chol", "thalach", "oldpeak"] if feature in df.columns]
gam_categorical_features = [feature for feature in ["sex", "cp", "fbs", "restecg", "exang", "slope"] if feature in df.columns]
gam_features = gam_numeric_features + gam_categorical_features
X_gam = df[gam_features].copy()
for feature in gam_numeric_features:
    X_gam[feature] = pd.to_numeric(X_gam[feature], errors="coerce").astype(float)
for feature in gam_categorical_features:
    encoded_values = X_gam[feature].map(CATEGORICAL_ENCODERS[feature])
    mode_value = _safe_mode(encoded_values)
    X_gam[feature] = encoded_values.fillna(mode_value).astype(int)
Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_gam, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


def build_model_gam():
    """Return a LogisticGAM model when pygam is available."""
    if LogisticGAM is None or s is None or f is None:
        return None

    terms = None
    for idx, feature in enumerate(gam_features):
        term = s(idx) if feature in gam_numeric_features else f(idx)
        terms = term if terms is None else terms + term
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
    "gam_numeric_features",
    "gam_categorical_features",
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
