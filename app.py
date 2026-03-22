from __future__ import annotations

from dash import Dash, Input, Output, dcc, html, dash_table, ALL, State, ctx
import dash
import json

from data import load_data
from menu import CATEGORICAL_BADGE_LABELS, badge_class, build_patient_input_panel
from models.rf import model_rf, rf_accuracy, rf_auc, gam_accuracy, gam_auc
from population import (
    SIMILAR_PATIENT_COLUMNS,
    build_population_context,
    make_population_summary_alert,
)
from shap_card import (
    get_default_patient_input,
    build_patient_dataframe,
    compute_prediction_and_shap,
    make_shap_figure,
)
from gam_card import build_gam_dataframe, predict_gam_risk, make_gam_figure
import dash_bootstrap_components as dbc



data = load_data()
default_patient_input = get_default_patient_input()


def risk_color(probability):
    probability = max(0.0, min(1.0, float(probability)))
    red = int(255 * probability)
    green = int(160 * (1 - probability))
    return f"rgb({red}, {green}, 60)"


def build_clinical_summary(model, patient_df, probability, gam_probability):
    try:
        _, _, shap_df = compute_prediction_and_shap(model, patient_df)
    except RuntimeError as exc:
        return [
            html.H5("Clinical Summary", className="mb-3"),
            html.Div(str(exc), className="text-muted"),
        ]

    drivers = shap_df[shap_df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(3)
    protective = shap_df[shap_df["shap_value"] < 0].sort_values("shap_value", ascending=True).head(2)

    summary = [
        html.H5("Clinical Summary", className="mb-3"),
        html.Div(f"Random Forest predicted probability: {probability:.1%}", className="mb-1"),
    ]
    if gam_probability is not None:
        summary.append(html.Div(f"GAM predicted probability: {gam_probability:.1%}", className="mb-3"))
    else:
        summary.append(html.Div("GAM predicted probability unavailable", className="mb-3"))

    summary.append(html.Div("Top 3 Drivers", className="fw-semibold mb-2"))
    if drivers.empty:
        summary.append(html.Div("No positive drivers identified for this patient.", className="text-muted mb-3"))
    else:
        summary.append(html.Ul([
            html.Li(f"{row.label} ({row.value})") for row in drivers.itertuples()
        ], className="mb-3"))

    summary.append(html.Div("Top 2 Protective", className="fw-semibold mb-2"))
    if protective.empty:
        summary.append(html.Div("No protective factors identified for this patient.", className="text-muted"))
    else:
        summary.append(html.Ul([
            html.Li(f"{row.label} ({row.value})") for row in protective.itertuples()
        ], className="mb-0"))

    return summary


initial_shap_figure = make_shap_figure(
    model_rf,
    build_patient_dataframe(default_patient_input),
)
initial_gam_figure = make_gam_figure(build_gam_dataframe(default_patient_input))
initial_rf_probability = float(
    model_rf.predict_proba(build_patient_dataframe(default_patient_input))[0, 1]
)
initial_gam_probability = predict_gam_risk(build_gam_dataframe(default_patient_input))
initial_clinical_summary = build_clinical_summary(
    model_rf,
    build_patient_dataframe(default_patient_input),
    initial_rf_probability,
    initial_gam_probability,
)
initial_population_summary, initial_similar_patients, initial_population_figure = build_population_context(default_patient_input)

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
                build_patient_input_panel(data, default_patient_input),
                dbc.Col([
                    html.H4("SHAP Explanation", className="mb-4"),
                    dcc.Graph(id="shap-patient-graph", figure=initial_shap_figure, config={"displayModeBar": False}),
                ], width=6),
                dbc.Col([
                    dbc.Alert(
                        initial_clinical_summary,
                        id="prediction-summary",
                        color="primary",
                        className="mb-0 h-100",
                    ),
                ], width=3),
            ])
        ])
    ], className='mb-4'),
    dbc.Card([
        dbc.CardHeader(html.H2("Population Context", className="mb-0"), className='bg-light'),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    make_population_summary_alert(initial_population_summary),
                    dash_table.DataTable(
                        id="similar-patients-table",
                        columns=SIMILAR_PATIENT_COLUMNS,
                        data=initial_similar_patients,
                        page_size=8,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "fontSize": "0.9rem", "padding": "0.5rem"},
                        style_header={"fontWeight": "600"},
                    ),
                ], width=4),
                dbc.Col([
                    dcc.Graph(
                        id="population-distribution-graph",
                        figure=initial_population_figure,
                        config={"displayModeBar": False},
                    ),
                ], width=8),
            ], className="g-4"),
        ])
    ], className='mb-4'),
    
    dbc.Card([
        dbc.CardHeader(html.H2("Feature Effects", className="mb-0"), className='bg-light'),
        dbc.CardBody([
            html.H4("GAM Component", className="mb-4"),
            dcc.Graph(id="gam-patient-graph", figure=initial_gam_figure, config={"displayModeBar": False}),
        ])
    ], className='mb-4'),



], fluid=True)

# ========= Callbacks ===========


def register_categorical_badge_callback(field_name):
    values = sorted(data[field_name].dropna().unique().tolist())

    @app.callback(
        Output(f"input-{field_name}", "data"),
        [Output({"type": f"{field_name}-badge", "value": value}, "className") for value in values],
        [Input({"type": f"{field_name}-badge", "value": value}, "n_clicks") for value in values],
        prevent_initial_call=False,
    )
    def update_categorical_badges(*_):
        triggered = ctx.triggered_id
        selected = default_patient_input[field_name] if triggered is None else triggered["value"]
        classes = [
            badge_class(value == selected, "me-2 mb-2")
            for value in values
        ]
        return (selected, *classes)

    return update_categorical_badges


for categorical_field in CATEGORICAL_BADGE_LABELS:
    register_categorical_badge_callback(categorical_field)


# @app.callback(
#     Output('pcp-selected-badges', 'children'),
#     Input('pcp-feature-selector', 'value')
# )
# def update_pcp_badges(selected_features):
#     return get_pcp_badges(selected_features)


# @app.callback(
#     Output('pcp-feature-selector', 'value'),
#     Input({'type': 'pcp-dim-badge', 'index': ALL}, 'n_clicks'),
#     State('pcp-feature-selector', 'value')
# )
# def sync_pcp_feature_with_badges(n_clicks, selected_features):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         return selected_features

#     triggered = ctx.triggered[0]['prop_id'].split('.')[0]
#     try:
#         triggered_obj = json.loads(triggered)
#     except ValueError:
#         return selected_features

#     clicked_value = triggered_obj.get('index')
#     return toggle_pcp_feature(clicked_value, selected_features)


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
    Output("population-summary", "children"),
    Output("similar-patients-table", "data"),
    Output("population-distribution-graph", "figure"),
    Output("gam-patient-graph", "figure"),
    Input("input-age", "value"),
    Input("input-sex", "data"),
    Input("input-cp", "data"),
    Input("input-trestbps", "value"),
    Input("input-chol", "value"),
    Input("input-fbs", "data"),
    Input("input-restecg", "data"),
    Input("input-thalach", "value"),
    Input("input-exang", "data"),
    Input("input-oldpeak", "value"),
    Input("input-slope", "data"),
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

    agreement = "Unavailable"
    if gam_probability is not None:
        rf_positive = probability >= 0.5
        gam_positive = gam_probability >= 0.5
        agreement = "Agree" if rf_positive == gam_positive else "Disagree"

    model_metrics = f"{rf_auc:.2f} / {rf_accuracy:.2f}"
    summary = build_clinical_summary(model_rf, patient_df, probability, gam_probability)
    population_summary, similar_patients, population_figure = build_population_context(patient_input)
    return (
        f"{probability:.1%}",
        {"color": risk_color(probability)},
        risk_category,
        agreement,
        model_metrics,
        summary,
        make_shap_figure(model_rf, patient_df),
        population_summary,
        similar_patients,
        population_figure,
        make_gam_figure(gam_df),
    )

# ======== Run App ========
if __name__ == "__main__":
    app.run(debug=True)
