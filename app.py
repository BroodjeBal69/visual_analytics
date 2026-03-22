from __future__ import annotations

from dash import Dash, Input, Output, dcc, html, dash_table, ALL, State
import dash
import json

from data import load_data
from models.rf import model_rf, rf_accuracy, rf_auc, gam_accuracy, gam_auc
from pcp import FEATURE_OPTIONS, render_pcp, get_pcp_badges, toggle_pcp_feature
from shap_card import get_default_patient_input, build_patient_dataframe, make_shap_figure
from gam_card import build_gam_dataframe, predict_gam_risk, make_gam_figure
import dash_bootstrap_components as dbc



# ======== Initialization ========
# Load data
data = load_data()
default_patient_input = get_default_patient_input()


def categorical_options(column_name):
    values = sorted(data[column_name].dropna().unique().tolist())
    return [{"label": str(value), "value": value} for value in values]


def slider_marks(column_name):
    min_value = float(data[column_name].min())
    max_value = float(data[column_name].max())

    def format_value(value):
        return str(int(value)) if float(value).is_integer() else f"{value:.1f}"

    return {
        min_value: format_value(min_value),
        max_value: format_value(max_value),
    }


def risk_color(probability):
    probability = max(0.0, min(1.0, float(probability)))
    red = int(255 * probability)
    green = int(160 * (1 - probability))
    return f"rgb({red}, {green}, 60)"


initial_shap_figure = make_shap_figure(
    model_rf,
    build_patient_dataframe(default_patient_input),
)
initial_gam_figure = make_gam_figure(build_gam_dataframe(default_patient_input))
initial_rf_probability = float(
    model_rf.predict_proba(build_patient_dataframe(default_patient_input))[0, 1]
)
initial_gam_probability = predict_gam_risk(build_gam_dataframe(default_patient_input))

# Get model performance metrics
performance_metrics = {
    "Random Forest Accuracy": rf_accuracy,
    "Random Forest AUC": rf_auc,
    "GAM Accuracy": gam_accuracy,
    "GAM AUC": gam_auc,
}

# ========= App ===========

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# ========= Layout ===========
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),

    dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H1("Heart Disease Risk Dashboard", className="mb-1 p-3 bg-info-subtle text-info-emphasis text-center fw-bold"),
                    html.Div(
                        "For patient-specific risk prediction and model insights",
                        className="card-title, text-muted text-center text-lighter"
                    ),
                ],
                className="p-3",
            ),
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Predicted Risk", className="text-muted small"),
                                html.H2(
                                    f"{initial_rf_probability:.1%}",
                                    id="predicted-risk-value",
                                    className="mb-0",
                                    style={"color": risk_color(initial_rf_probability)},
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Div("Risk Category", className="text-muted small"),
                                html.H2(
                                    "Moderate" if initial_rf_probability >= 0.4 else "Low",
                                    id="risk-category-value",
                                    className="mb-0",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Div("RF vs GAM", className="text-muted small"),
                                html.H2(
                                    "Agree" if initial_gam_probability is not None else "Unavailable",
                                    id="rf-gam-agreement",
                                    className="mb-0",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Div("AUC / Accuracy", className="text-muted small"),
                                html.H2(
                                    f"{rf_auc:.2f} / {rf_accuracy:.2f}",
                                    id="model-metrics-value",
                                    className="mb-0",
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="g-3 align-items-center",
                )
            ),
        ],
        className="mb-4 shadow-sm",
    ),

    
    # SHAP card
    dbc.Card([
        dbc.CardHeader(html.H2("Risk Predictor", className="mb-0"), className='bg-light'),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4("Model Input", className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Age"),
                            dcc.Slider(
                                id="input-age",
                                min=int(data["age"].min()),
                                max=int(data["age"].max()),
                                step=1,
                                value=int(default_patient_input["age"]),
                                marks=slider_marks("age"),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Sex"),
                            dbc.Select(id="input-sex", options=categorical_options("sex"), value=default_patient_input["sex"]),
                        ], width=6),
                    ], className="g-3 mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Chest Pain Type"),
                            dbc.Select(id="input-cp", options=categorical_options("cp"), value=default_patient_input["cp"]),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Fasting Blood Sugar"),
                            dbc.Select(id="input-fbs", options=categorical_options("fbs"), value=default_patient_input["fbs"]),
                        ], width=6),
                    ], className="g-3 mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Resting BP"),
                            dcc.Slider(
                                id="input-trestbps",
                                min=float(data["trestbps"].min()),
                                max=float(data["trestbps"].max()),
                                step=1,
                                value=float(default_patient_input["trestbps"]),
                                marks=slider_marks("trestbps"),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Cholesterol"),
                            dcc.Slider(
                                id="input-chol",
                                min=float(data["chol"].min()),
                                max=float(data["chol"].max()),
                                step=1,
                                value=float(default_patient_input["chol"]),
                                marks=slider_marks("chol"),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], width=6),
                    ], className="g-3 mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Resting ECG"),
                            dbc.Select(id="input-restecg", options=categorical_options("restecg"), value=default_patient_input["restecg"]),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Max Heart Rate"),
                            dcc.Slider(
                                id="input-thalach",
                                min=float(data["thalach"].min()),
                                max=float(data["thalach"].max()),
                                step=1,
                                value=float(default_patient_input["thalach"]),
                                marks=slider_marks("thalach"),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], width=6),
                    ], className="g-3 mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Exercise Angina"),
                            dbc.Select(id="input-exang", options=categorical_options("exang"), value=default_patient_input["exang"]),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("ST Depression"),
                            dcc.Slider(
                                id="input-oldpeak",
                                min=float(data["oldpeak"].min()),
                                max=float(data["oldpeak"].max()),
                                step=0.1,
                                value=float(default_patient_input["oldpeak"]),
                                marks=slider_marks("oldpeak"),
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], width=6),
                    ], className="g-3 mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("ST Slope"),
                            dbc.Select(id="input-slope", options=categorical_options("slope"), value=default_patient_input["slope"]),
                        ], width=6),
                    ], className="g-3 mb-4"),
                    dbc.Alert(id="prediction-summary", color="primary", className="mb-0"),
                ], width=4),
                dbc.Col([
                    html.H4("SHAP Explanation", className="mb-4"),
                    dcc.Graph(id="shap-patient-graph", figure=initial_shap_figure, config={"displayModeBar": False}),
                    html.Hr(className="my-4"),
                    html.H4("GAM Component", className="mb-4"),
                    dcc.Graph(id="gam-patient-graph", figure=initial_gam_figure, config={"displayModeBar": False}),
                ], width=8),
            ])
        ])
    ], className='mb-4'),



], fluid=True)

# ========= Callbacks ===========

@app.callback(
    Output('pcp-selected-badges', 'children'),
    Input('pcp-feature-selector', 'value')
)
def update_pcp_badges(selected_features):
    return get_pcp_badges(selected_features)


@app.callback(
    Output('pcp-feature-selector', 'value'),
    Input({'type': 'pcp-dim-badge', 'index': ALL}, 'n_clicks'),
    State('pcp-feature-selector', 'value')
)
def sync_pcp_feature_with_badges(n_clicks, selected_features):
    ctx = dash.callback_context
    if not ctx.triggered:
        return selected_features

    triggered = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        triggered_obj = json.loads(triggered)
    except ValueError:
        return selected_features

    clicked_value = triggered_obj.get('index')
    return toggle_pcp_feature(clicked_value, selected_features)


@app.callback(
    [Output('parallel-coords-chart', 'figure'), Output('patient-data-table', 'data')],
    [Input('pcp-feature-selector', 'value'),
     Input('age-range-slider', 'value'),
     Input('target-filter', 'value')]
)
def update_patient_explorer(selected_features, age_range, target_values):
    return render_pcp(data, selected_features, age_range, target_values)


@app.callback(
    Output("predicted-risk-value", "children"),
    Output("predicted-risk-value", "style"),
    Output("risk-category-value", "children"),
    Output("rf-gam-agreement", "children"),
    Output("model-metrics-value", "children"),
    Output("prediction-summary", "children"),
    Output("shap-patient-graph", "figure"),
    Output("gam-patient-graph", "figure"),
    Input("input-age", "value"),
    Input("input-sex", "value"),
    Input("input-cp", "value"),
    Input("input-trestbps", "value"),
    Input("input-chol", "value"),
    Input("input-fbs", "value"),
    Input("input-restecg", "value"),
    Input("input-thalach", "value"),
    Input("input-exang", "value"),
    Input("input-oldpeak", "value"),
    Input("input-slope", "value"),
)
def update_shap_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope):
    patient_input = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
    }
    patient_df = build_patient_dataframe(patient_input)
    gam_df = build_gam_dataframe(patient_input)
    probability = float(model_rf.predict_proba(patient_df)[0, 1])
    gam_probability = predict_gam_risk(gam_df)
    if probability >= 0.7:
        risk_category = "High"
    elif probability >= 0.4:
        risk_category = "Moderate"
    else:
        risk_category = "Low"

    prediction = f"{risk_category} risk of heart disease"

    agreement = "Unavailable"
    if gam_probability is not None:
        rf_positive = probability >= 0.5
        gam_positive = gam_probability >= 0.5
        agreement = "Agree" if rf_positive == gam_positive else "Disagree"

    summary = [
        html.H5(prediction, className="mb-1"),
        html.Div(f"Random Forest predicted probability: {probability:.1%}"),
    ]
    if gam_probability is not None:
        summary.append(html.Div(f"GAM predicted probability: {gam_probability:.1%}"))
    model_metrics = f"{rf_auc:.2f} / {rf_accuracy:.2f}"
    return (
        f"{probability:.1%}",
        {"color": risk_color(probability)},
        risk_category,
        agreement,
        model_metrics,
        summary,
        make_shap_figure(model_rf, patient_df),
        make_gam_figure(gam_df),
    )

# ======== Run App ========
if __name__ == "__main__":
    app.run(debug=True)
