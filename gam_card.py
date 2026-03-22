import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import (
    CATEGORICAL_ENCODERS,
    cp_map,
    exang_map,
    fbs_map,
    label_map,
    load_data,
    restecg_map,
    sex_map,
    slope_map,
)
from models.rf import gam_categorical_features, gam_features, gam_numeric_features, model_gam


GAM_CATEGORY_LABELS = {
    "sex": sex_map,
    "cp": cp_map,
    "fbs": fbs_map,
    "restecg": restecg_map,
    "exang": exang_map,
    "slope": slope_map,
}
GAM_REFERENCE_DATA = load_data()[gam_features]
GAM_CATEGORICAL_DEFAULTS = {}


def _safe_numeric_mode(series: pd.Series, fallback: int = 0) -> int:
    cleaned = series.dropna()
    if cleaned.empty:
        return fallback
    modes = cleaned.mode()
    if modes.empty:
        return fallback
    return int(modes.iloc[0])


for feature in gam_numeric_features:
    GAM_REFERENCE_DATA[feature] = pd.to_numeric(GAM_REFERENCE_DATA[feature], errors="coerce").astype(float)
for feature in gam_categorical_features:
    numeric_values = GAM_REFERENCE_DATA[feature].map(CATEGORICAL_ENCODERS[feature])
    default_value = _safe_numeric_mode(numeric_values)
    GAM_CATEGORICAL_DEFAULTS[feature] = default_value
    GAM_REFERENCE_DATA[feature] = numeric_values.fillna(default_value).astype(int)


def _coerce_gam_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df[gam_features].copy()
    for feature in gam_numeric_features:
        coerced[feature] = pd.to_numeric(coerced[feature], errors="coerce").astype(float)
    for feature in gam_categorical_features:
        coerced[feature] = (
            coerced[feature].map(CATEGORICAL_ENCODERS[feature])
            .fillna(GAM_CATEGORICAL_DEFAULTS[feature])
            .astype(int)
        )
    return coerced


def _gam_reference_row() -> pd.DataFrame:
    reference_values = {}
    for feature in gam_features:
        series = GAM_REFERENCE_DATA[feature].dropna()
        if feature in gam_categorical_features:
            reference_values[feature] = _safe_numeric_mode(series, GAM_CATEGORICAL_DEFAULTS[feature])
        else:
            reference_values[feature] = float(series.median())
    return _coerce_gam_dataframe(pd.DataFrame([reference_values], columns=gam_features))


def _categorical_effect_frame(feature: str, term_index: int) -> pd.DataFrame:
    reference_df = _gam_reference_row()
    values = sorted(GAM_REFERENCE_DATA[feature].dropna().unique().tolist())
    rows = []
    for value in values:
        scenario_df = reference_df.copy()
        scenario_df.loc[0, feature] = value
        effect = float(model_gam.partial_dependence(term=term_index, X=scenario_df.to_numpy(dtype=float))[0])
        rows.append(
            {
                "value": value,
                "label": GAM_CATEGORY_LABELS.get(feature, {}).get(value, str(value)),
                "effect": effect,
            }
        )
    return pd.DataFrame(rows)


def build_gam_dataframe(patient_input: dict) -> pd.DataFrame:
    """Return the subset of fields used by the GAM model."""
    row = {feature: patient_input.get(feature) for feature in gam_features}
    return _coerce_gam_dataframe(pd.DataFrame([row]))


def predict_gam_risk(patient_df: pd.DataFrame):
    """Return the GAM risk probability for a single patient."""
    if model_gam is None:
        return None
    gam_df = _coerce_gam_dataframe(patient_df)
    return float(model_gam.predict_proba(gam_df.to_numpy(dtype=float))[0])


def make_gam_figure(patient_df: pd.DataFrame) -> go.Figure:
    """Plot GAM partial effects with the patient's current values highlighted."""
    if model_gam is None:
        fig = go.Figure()
        fig.add_annotation(
            text="GAM explanation unavailable. Install `pygam` to enable this panel.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15},
        )
        fig.update_layout(
            title="GAM partial effects unavailable",
            xaxis={"visible": False},
            yaxis={"visible": False},
            template="plotly_white",
            height=520,
        )
        return fig

    patient_df = _coerce_gam_dataframe(patient_df)

    cols = 2
    rows = math.ceil(len(gam_features) / cols)
    subplot_titles = [label_map.get(feature, feature.title()) for feature in gam_features]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, vertical_spacing=0.14)

    for idx, feature in enumerate(gam_features):
        row = idx // cols + 1
        col = idx % cols + 1

        if feature in gam_numeric_features:
            grid = model_gam.generate_X_grid(term=idx)
            partial_effect = model_gam.partial_dependence(term=idx, X=grid)
            patient_value = float(patient_df.iloc[0][feature])
            closest_idx = int(np.abs(grid[:, idx] - patient_value).argmin())
            patient_effect = float(partial_effect[closest_idx])

            fig.add_trace(
                go.Scatter(
                    x=grid[:, idx],
                    y=partial_effect,
                    mode="lines",
                    line={"color": "#0d6efd", "width": 2},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[patient_value],
                    y=[patient_effect],
                    mode="markers",
                    marker={"color": "#dc3545", "size": 10},
                    showlegend=False,
                    hovertemplate=f"{label_map.get(feature, feature)}: %{{x:.2f}}<br>Effect: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        else:
            categorical_effects = _categorical_effect_frame(feature, idx)
            patient_value = patient_df.iloc[0][feature]
            colors = [
                "#dc3545" if value == patient_value else "#0d6efd"
                for value in categorical_effects["value"]
            ]
            fig.add_trace(
                go.Bar(
                    x=categorical_effects["label"],
                    y=categorical_effects["effect"],
                    marker_color=colors,
                    showlegend=False,
                    hovertemplate=f"{label_map.get(feature, feature)}: %{{x}}<br>Effect: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text=label_map.get(feature, feature.title()), row=row, col=col)
        fig.update_yaxes(title_text="Partial effect", row=row, col=col)

    fig.update_layout(
        title="GAM Feature Effects for Current Patient",
        template="plotly_white",
        height=260 * rows,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig
