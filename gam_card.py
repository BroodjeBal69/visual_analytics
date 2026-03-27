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
from palette import (
    NEGATIVE_COLOR,
    NEUTRAL_LINE_COLOR,
    NEUTRAL_TEXT_COLOR,
    PATIENT_COLOR,
    POSITIVE_COLOR,
)

LOW_RISK_BG = "rgba(99, 159, 194, 0.18)"
HIGH_RISK_BG = "rgba(221, 124, 124, 0.18)"
GAM_MIN_HEIGHT = 400
GAM_MAX_HEIGHT = 1000


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

# Precompute reference data and default values for categorical features based on the dataset.
def _safe_numeric_mode(series: pd.Series, fallback: int = 0) -> int:
    cleaned = series.dropna()
    if cleaned.empty:
        return fallback
    modes = cleaned.mode()
    if modes.empty:
        return fallback
    return int(modes.iloc[0])

# Coerce GAM reference data to numeric and compute defaults for categorical features
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

# ====== Build reference row for GAM partial dependence calculations ======
def _gam_reference_row() -> pd.DataFrame:
    reference_values = {}
    for feature in gam_features:
        series = GAM_REFERENCE_DATA[feature].dropna()
        if feature in gam_categorical_features:
            reference_values[feature] = _safe_numeric_mode(series, GAM_CATEGORICAL_DEFAULTS[feature])
        else:
            reference_values[feature] = float(series.median())
    return _coerce_gam_dataframe(pd.DataFrame([reference_values], columns=gam_features))

# ====== Build categorical effect frame for GAM features ======
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

# ====== Patient dataframe for GAM predictions and explanations ======
def build_gam_dataframe(patient_input: dict) -> pd.DataFrame:
    """Return the subset of fields used by the GAM model."""
    row = {feature: patient_input.get(feature) for feature in gam_features}
    return _coerce_gam_dataframe(pd.DataFrame([row]))

# Predicts the risk for a single patient
def predict_gam_risk(patient_df: pd.DataFrame):
    """Return the GAM risk probability for a single patient."""
    if model_gam is None:
        return None
    gam_df = _coerce_gam_dataframe(patient_df)
    return float(model_gam.predict_proba(gam_df.to_numpy(dtype=float))[0])

# Plots partial effects of GAM
def make_gam_figure(patient_df: pd.DataFrame, selected_features: list[str] | None = None) -> go.Figure:
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
    features_to_plot = [feature for feature in gam_features if selected_features is None or feature in selected_features]

    if not features_to_plot:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one GAM feature to display.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15},
        )
        fig.update_layout(
            title="GAM Feature Effects for Current Patient",
            xaxis={"visible": False},
            yaxis={"visible": False},
            template="plotly_white",
            height=260,
        )
        return fig

    cols = 2
    rows = math.ceil(len(features_to_plot) / cols)
    subplot_titles = [label_map.get(feature, feature.title()) for feature in features_to_plot]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, vertical_spacing=0.14)

    for plot_idx, feature in enumerate(features_to_plot):
        row = plot_idx // cols + 1
        col = plot_idx % cols + 1
        term_index = gam_features.index(feature)

        if feature in gam_numeric_features:
            grid = model_gam.generate_X_grid(term=term_index)
            partial_effect = model_gam.partial_dependence(term=term_index, X=grid)
            patient_value = float(patient_df.iloc[0][feature])
            closest_idx = int(np.abs(grid[:, term_index] - patient_value).argmin())
            patient_effect = float(partial_effect[closest_idx])
            y_min = float(np.min(partial_effect))
            y_max = float(np.max(partial_effect))

            if y_min < 0:
                fig.add_hrect(
                    y0=y_min,
                    y1=0,
                    fillcolor=LOW_RISK_BG,
                    line_width=0,
                    layer="below",
                    row=row,
                    col=col,
                )
            if y_max > 0:
                fig.add_hrect(
                    y0=0,
                    y1=y_max,
                    fillcolor=HIGH_RISK_BG,
                    line_width=0,
                    layer="below",
                    row=row,
                    col=col,
                )

            fig.add_trace(
                go.Scatter(
                    x=grid[:, term_index],
                    y=partial_effect,
                    mode="lines",
                    line={"color": NEGATIVE_COLOR, "width": 2},
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
                    marker={
                        "color": PATIENT_COLOR,
                        "size": 10,
                    },
                    showlegend=False,
                    hovertemplate=f"{label_map.get(feature, feature)}: %{{x:.2f}}<br>Effect: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        else:
            categorical_effects = _categorical_effect_frame(feature, term_index)
            patient_value = patient_df.iloc[0][feature]
            y_min = float(categorical_effects["effect"].min())
            y_max = float(categorical_effects["effect"].max())

            if y_min < 0:
                fig.add_hrect(
                    y0=y_min,
                    y1=0,
                    fillcolor=LOW_RISK_BG,
                    line_width=0,
                    layer="below",
                    row=row,
                    col=col,
                )
            if y_max > 0:
                fig.add_hrect(
                    y0=0,
                    y1=y_max,
                    fillcolor=HIGH_RISK_BG,
                    line_width=0,
                    layer="below",
                    row=row,
                    col=col,
                )

            colors = [
                POSITIVE_COLOR if effect >= 0 else NEGATIVE_COLOR
                for effect in categorical_effects["effect"]
            ]
            line_widths = [
                3 if value == patient_value else 0
                for value in categorical_effects["value"]
            ]
            line_colors = [
                PATIENT_COLOR if value == patient_value else "rgba(0,0,0,0)"
                for value in categorical_effects["value"]
            ]
            fig.add_trace(
                go.Bar(
                    x=categorical_effects["label"],
                    y=categorical_effects["effect"],
                    marker={
                        "color": colors,
                        "line": {"color": line_colors, "width": line_widths},
                    },
                    showlegend=False,
                    hovertemplate=f"{label_map.get(feature, feature)}: %{{x}}<br>Effect: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(
            title_text=label_map.get(feature, feature.title()),
            showline=True,
            linecolor=NEUTRAL_LINE_COLOR,
            tickfont={"color": NEUTRAL_TEXT_COLOR},
            title_font={"color": NEUTRAL_TEXT_COLOR},
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Partial effect",
            showline=True,
            linecolor=NEUTRAL_LINE_COLOR,
            zeroline=True,
            zerolinecolor=NEUTRAL_LINE_COLOR,
            tickfont={"color": NEUTRAL_TEXT_COLOR},
            title_font={"color": NEUTRAL_TEXT_COLOR},
            row=row,
            col=col,
        )

    fig.update_layout(
        title="GAM Feature Effects for Current Patient",
        template="plotly_white",
        height=min(GAM_MAX_HEIGHT, max(GAM_MIN_HEIGHT, 200 * rows)),
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig
