import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import label_map
from models.rf import gam_features, model_gam


def build_gam_dataframe(patient_input: dict) -> pd.DataFrame:
    """Return the subset of fields used by the GAM model."""
    row = {feature: patient_input.get(feature) for feature in gam_features}
    return pd.DataFrame([row])


def predict_gam_risk(patient_df: pd.DataFrame):
    """Return the GAM risk probability for a single patient."""
    if model_gam is None:
        return None
    return float(model_gam.predict_proba(patient_df[gam_features])[0])


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

    cols = 2
    rows = math.ceil(len(gam_features) / cols)
    subplot_titles = [label_map.get(feature, feature.title()) for feature in gam_features]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, vertical_spacing=0.14)

    for idx, feature in enumerate(gam_features):
        grid = model_gam.generate_X_grid(term=idx)
        partial_effect = model_gam.partial_dependence(term=idx, X=grid)
        patient_value = float(patient_df.iloc[0][feature])
        closest_idx = int(np.abs(grid[:, idx] - patient_value).argmin())
        patient_effect = float(partial_effect[closest_idx])

        row = idx // cols + 1
        col = idx % cols + 1

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
        fig.update_xaxes(title_text=label_map.get(feature, feature.title()), row=row, col=col)
        fig.update_yaxes(title_text="Partial effect", row=row, col=col)

    fig.update_layout(
        title="GAM Partial Effects for Current Patient",
        template="plotly_white",
        height=260 * rows,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig
