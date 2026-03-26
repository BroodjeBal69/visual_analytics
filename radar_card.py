import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from data import label_map
from models.rf import X_train, available_features, categorical_features, numeric_features
from palette import NEGATIVE_COLOR, PATIENT_COLOR


def _clip_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _build_training_scales() -> tuple[dict, dict]:
    """Build normalization scales from training data once per figure call."""
    numeric_scales = {}
    for col in numeric_features:
        if col in X_train.columns:
            min_val = float(X_train[col].min())
            max_val = float(X_train[col].max())
            numeric_scales[col] = (min_val, max_val)

    categorical_scales = {}
    for col in categorical_features:
        if col in X_train.columns:
            values = sorted(X_train[col].dropna().unique().tolist())
            if len(values) <= 1:
                categorical_scales[col] = {values[0]: 0.5} if values else {}
            else:
                denom = float(len(values) - 1)
                categorical_scales[col] = {v: i / denom for i, v in enumerate(values)}

    return numeric_scales, categorical_scales


def _normalize_features(X_df: pd.DataFrame, numeric_scales: dict, categorical_scales: dict) -> pd.DataFrame:
    """Normalize features to [0, 1] using scales derived from training data."""
    normalized = X_df.copy()

    for col in numeric_features:
        if col not in X_df.columns or col not in numeric_scales:
            continue
        min_val, max_val = numeric_scales[col]
        if max_val > min_val:
            normalized[col] = ((X_df[col].astype(float) - min_val) / (max_val - min_val)).clip(0.0, 1.0)
        else:
            normalized[col] = 0.5

    for col in categorical_features:
        if col not in X_df.columns:
            continue
        mapping = categorical_scales.get(col, {})
        normalized[col] = X_df[col].map(mapping).fillna(0.5)

    return normalized


def _get_population_avg():
    """Get average (median) normalized feature values from training data."""
    numeric_scales, categorical_scales = _build_training_scales()
    normalized_train = _normalize_features(X_train, numeric_scales, categorical_scales)
    pop_avg = {}
    for feature in available_features:
        pop_avg[feature] = float(normalized_train[feature].median())
    return pop_avg


def make_radar_figure(patient_input: dict) -> go.Figure:
    """
    Create a radar chart comparing patient profile to population average.
    
    Args:
        patient_input: dict with feature values for the patient
        
    Returns:
        Plotly Figure object (radar chart)
    """
    # Get population averages
    pop_avg = _get_population_avg()
    
    # Normalize patient values using training-data scales
    patient_df = pd.DataFrame([{f: patient_input.get(f) for f in available_features}])
    numeric_scales, categorical_scales = _build_training_scales()
    normalized_patient = _normalize_features(patient_df, numeric_scales, categorical_scales)
    
    # Select features for display (exclude if all zeros or target)
    display_features = [f for f in available_features if f != "target"]
    
    # Get display labels
    categories = [label_map.get(f, f) for f in display_features]
    
    # Get normalized values for patient and population
    patient_values = [_clip_01(normalized_patient[f].iloc[0]) for f in display_features]
    population_values = [_clip_01(pop_avg[f]) for f in display_features]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add population average trace
    fig.add_trace(go.Scatterpolar(
        r=population_values,
        theta=categories,
        fill='toself',
        name='Population Average',
        line=dict(color=NEGATIVE_COLOR),
        fillcolor='rgba(99, 159, 194, 0.2)',
        hovertemplate='<b>%{theta}</b><br>Population: %{r:.2f}<extra></extra>',
    ))
    
    # Add patient trace
    fig.add_trace(go.Scatterpolar(
        r=patient_values,
        theta=categories,
        fill='toself',
        name='This Patient',
        line=dict(color=PATIENT_COLOR),
        fillcolor='rgba(171, 136, 216, 0.3)',
        hovertemplate='<b>%{theta}</b><br>Patient: %{r:.2f}<extra></extra>',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=11),
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.15,
            xanchor='center',
            yanchor='top',
            orientation='h',
            font=dict(size=12),
        ),
        autosize=True,
        margin=dict(l=100, r=100, t=80, b=120),
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
    )
    
    return fig


def build_radar_component():
    """Build the radar chart component for patient view."""
    return html.Div(
        [
            html.H6("Patient Profile Comparison", className="mb-3"),
            dcc.Graph(
                id="patient-radar-chart",
                config={"displayModeBar": False},
                style={"height": "420px"},
            ),
        ],
        className="mt-3",
    )
