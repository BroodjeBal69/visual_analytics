import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data import label_map, load_data
from models.rf import available_features, categorical_features
from palette import NEGATIVE_COLOR, NEUTRAL_LINE_COLOR, POSITIVE_COLOR


def get_default_patient_input() -> dict:
    """Build a sensible default patient profile from the dataset."""
    df = load_data()
    defaults = {}
    for feature in available_features:
        series = df[feature].dropna()
        if feature in categorical_features:
            defaults[feature] = series.mode().iloc[0]
        else:
            defaults[feature] = float(series.median())
    return defaults


def build_patient_dataframe(patient_input: dict) -> pd.DataFrame:
    """Convert the form values into a one-row dataframe for prediction."""
    row = {feature: patient_input.get(feature) for feature in available_features}
    return pd.DataFrame([row])


def _aggregate_shap_values(feature_names, shap_values, raw_columns):
    aggregated = {}
    for feature_name, shap_value in zip(feature_names, shap_values):
        if feature_name.startswith("num__"):
            raw_feature = feature_name.replace("num__", "", 1)
        elif feature_name.startswith("cat__"):
            encoded_feature = feature_name.replace("cat__", "", 1)
            raw_feature = next(
                (
                    column for column in raw_columns
                    if encoded_feature == column or encoded_feature.startswith(f"{column}_")
                ),
                encoded_feature.split("_", 1)[0],
            )
        else:
            raw_feature = feature_name
        aggregated[raw_feature] = aggregated.get(raw_feature, 0.0) + float(shap_value)
    return aggregated

# ====== SHAP Explanation Logic for Clinical Summary ======
def compute_prediction_and_shap(model, patient_df: pd.DataFrame):
    """Return prediction probability and feature-level SHAP values for one patient."""
    try:
        import shap
    except Exception as exc:
        raise RuntimeError(
            "SHAP dependencies are not available. Install `shap` and `matplotlib`."
        ) from exc

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(patient_df)
    probability = float(classifier.predict_proba(transformed)[0, 1])

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        row_values = np.asarray(shap_values[1])[0]
    else:
        row_values = np.asarray(shap_values)
        if row_values.ndim == 3:
            row_values = row_values[0, :, 1]
        else:
            row_values = row_values[0]

    transformed_feature_names = preprocessor.get_feature_names_out()
    aggregated_values = _aggregate_shap_values(
        transformed_feature_names,
        row_values,
        patient_df.columns.tolist(),
    )

    shap_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "label": label_map.get(feature, feature.replace("_", " ").title()),
                "value": patient_df.iloc[0][feature],
                "shap_value": aggregated_values.get(feature, 0.0),
            }
            for feature in patient_df.columns
        ]
    )
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=True)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        base_value = float(np.asarray(expected_value).reshape(-1)[-1])
    else:
        base_value = float(expected_value)

    return probability, base_value, shap_df

# ====== Build SHAP figure for clinical summary ======
def make_shap_figure(model, patient_df: pd.DataFrame) -> go.Figure:
    """Create a Plotly bar chart showing feature contributions."""
    try:
        probability, _, shap_df = compute_prediction_and_shap(model, patient_df)
    except RuntimeError as exc:
        fig = go.Figure()
        fig.add_annotation(
            text=str(exc),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15},
        )
        fig.update_layout(
            title="SHAP explanation unavailable",
            xaxis={"visible": False},
            yaxis={"visible": False},
            template="plotly_white",
        )
        return fig

    shap_df["display_value"] = shap_df["value"].astype(str)
    colors = np.where(shap_df["shap_value"] >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)

    fig = go.Figure(
        go.Bar(
            x=shap_df["shap_value"],
            y=shap_df["label"],
            orientation="h",
            marker_color=colors,
            customdata=np.stack([shap_df["display_value"]], axis=-1),
            hovertemplate="%{y}<br>Contribution: %{x:.4f}<br>Input: %{customdata[0]}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=NEUTRAL_LINE_COLOR)
    fig.update_layout(
        title=f"SHAP Contributions for Predicted Risk {probability:.1%}",
        xaxis_title="Impact on model output",
        yaxis_title="",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=420,
    )
    return fig
