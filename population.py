from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots

from data import (
    cp_map,
    exang_map,
    fbs_map,
    label_map,
    load_data,
    restecg_map,
    sex_map,
    slope_map,
    target_map,
)
from models.rf import available_features, model_rf

POSITIVE_COLOR = "#DD7C7C"
NEGATIVE_COLOR = "#8AB7D1"
PATIENT_COLOR = "#7f4bc4"


data = load_data()
DISPLAY_VALUE_MAPS = {
    "sex": sex_map,
    "cp": cp_map,
    "fbs": fbs_map,
    "restecg": restecg_map,
    "exang": exang_map,
    "slope": slope_map,
    "target": target_map,
}
SIMILARITY_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
SIMILARITY_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
POPULATION_CONTEXT_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
SIMILAR_PATIENT_COLUMNS = [
    {"name": "Age", "id": "age"},
    {"name": "Sex", "id": "sex"},
    {"name": "Chest Pain", "id": "cp"},
    {"name": "Exercise Angina", "id": "exang"},
    {"name": "Observed Outcome", "id": "target"},
]


def format_display_value(feature, value):
    if feature in DISPLAY_VALUE_MAPS:
        return DISPLAY_VALUE_MAPS[feature].get(value, str(value))
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def build_population_distribution_figure(patient_input, selected_features=None):
    features_to_plot = selected_features or POPULATION_CONTEXT_FEATURES
    if not features_to_plot:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one continuous variable to display.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15},
        )
        fig.update_layout(
            template="plotly_white",
            height=260,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            xaxis={"visible": False},
            yaxis={"visible": False},
            title="Where This Patient Sits in the Population",
        )
        return fig

    feature_titles = {
        "age": "Age Distribution",
        "trestbps": "Resting BP Distribution",
        "chol": "Cholesterol Distribution",
        "thalach": "Max Heart Rate Distribution",
        "oldpeak": "ST Depression Distribution",
    }
    fig = make_subplots(
        rows=1,
        cols=len(features_to_plot),
        subplot_titles=[feature_titles.get(feature, label_map.get(feature, feature.title())) for feature in features_to_plot],
    )
    risk_groups = [
        (0, "No disease", NEGATIVE_COLOR),
        (1, "Disease", POSITIVE_COLOR),
    ]

    for idx, feature in enumerate(features_to_plot, start=1):
        for target_value, label, color in risk_groups:
            subset = data.loc[data["target"] == target_value, feature]
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=label,
                    marker_color=color,
                    opacity=0.65,
                    showlegend=idx == 1,
                    nbinsx=18,
                ),
                row=1,
                col=idx,
            )
        fig.add_vline(
            x=float(patient_input[feature]),
            line_color=PATIENT_COLOR,
            line_width=3,
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text=label_map.get(feature, feature.title()), row=1, col=idx)
        fig.update_yaxes(title_text="Patients" if idx == 1 else "", row=1, col=idx)

    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        height=330,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        title="Where This Patient Sits in the Population",
    )
    return fig


def build_population_context(patient_input, selected_features=None):
    distance = 0
    for feature in SIMILARITY_NUMERIC_FEATURES:
        feature_std = float(data[feature].std()) or 1.0
        distance += (data[feature] - float(patient_input[feature])).abs() / feature_std
    for feature in SIMILARITY_CATEGORICAL_FEATURES:
        distance += (data[feature] != patient_input[feature]).astype(float)

    similar_patients = data.assign(_distance=distance).sort_values("_distance").head(8).copy()
    similar_rf_risk = float(model_rf.predict_proba(similar_patients[available_features])[:, 1].mean())
    similar_observed_rate = float(similar_patients["target"].mean())

    summary = [
        html.H5("Population Context", className="mb-3"),
        html.Div(f"Most similar cohort size: {len(similar_patients)} patients", className="mb-1"),
        html.Div(f"Average RF risk in similar patients: {similar_rf_risk:.1%}", className="mb-1"),
        html.Div(f"Observed disease rate in similar patients: {similar_observed_rate:.1%}", className="mb-0"),
    ]

    similar_records = [
        {
            "age": int(row.age),
            "sex": format_display_value("sex", row.sex),
            "cp": format_display_value("cp", row.cp),
            "exang": format_display_value("exang", row.exang),
            "target": format_display_value("target", row.target),
        }
        for row in similar_patients.itertuples()
    ]

    return summary, similar_records, build_population_distribution_figure(patient_input, selected_features)


def make_population_summary_alert(children, component_id: str = "population-summary"):
    return dbc.Alert(children, id=component_id, color="info", className="mb-3")
