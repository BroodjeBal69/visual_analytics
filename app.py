from __future__ import annotations

from dash import Dash, Input, Output, dcc, html, dash_table, ALL, State, ctx
import dash
import dash_bootstrap_components as dbc
import json

from gam_card import build_gam_dataframe, predict_gam_risk, make_gam_figure
from data import load_data
from data import label_map
from menu import CATEGORICAL_BADGE_LABELS, build_patient_input_panel, badge_class
from models.rf import model_rf, rf_accuracy, rf_auc, gam_accuracy, gam_auc
from population import POPULATION_CONTEXT_FEATURES, SIMILAR_PATIENT_COLUMNS, build_population_context, make_population_summary_alert
from rule_mining import (
    build_rule_detail_card,
    build_top_rules_table_data,
    filter_rules_by_antecedents,
    filter_rules_by_outcome,
    get_matrix_rules,
    get_rule_filter_options,
    make_rule_matrix_figure,
    build_plot_rules,
    RULES_DF,
)
from shap_card import get_default_patient_input,build_patient_dataframe,compute_prediction_and_shap, make_shap_figure
from gam_card import build_gam_dataframe, predict_gam_risk, make_gam_figure
from models.rf import gam_features


data = load_data()
default_patient_input = get_default_patient_input()

def risk_color(probability):
    probability = max(0.0, min(1.0, float(probability)))
    red = int(255 * probability)
    green = int(160 * (1 - probability))
    return f"rgb({red}, {green}, 60)"

def build_clinical_summary(model, patient_df, probability, gam_probability):
    pass
    # try:
    #     _, _, shap_df = compute_prediction_and_shap(model, patient_df)
    # except RuntimeError as exc:
    #     return [
    #         html.H5("Clinical Summary", className="mb-3"),
    #         html.Div(str(exc), className="text-muted"),
    #     ]

    # drivers = shap_df[shap_df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(3)
    # protective = shap_df[shap_df["shap_value"] < 0].sort_values("shap_value", ascending=True).head(2)

    # summary = [
    #     html.H5("Clinical Summary", className="mb-3"),
    #     html.Div(f"Random Forest predicted probability: {probability:.1%}", className="mb-1"),
    # ]
    # if gam_probability is not None:
    #     summary.append(html.Div(f"GAM predicted probability: {gam_probability:.1%}", className="mb-3"))
    # else:
    #     summary.append(html.Div("GAM predicted probability unavailable", className="mb-3"))

    # summary.append(html.Div("Top 3 Drivers", className="fw-semibold mb-2"))
    # if drivers.empty:
    #     summary.append(html.Div("No positive drivers identified for this patient.", className="text-muted mb-3"))
    # else:
    #     summary.append(html.Ul([
    #         html.Li(f"{row.label} ({row.value})") for row in drivers.itertuples()
    #     ], className="mb-3"))

    # summary.append(html.Div("Top 2 Protective", className="fw-semibold mb-2"))
    # if protective.empty:
    #     summary.append(html.Div("No protective factors identified for this patient.", className="text-muted"))
    # else:
    #     summary.append(html.Ul([
    #         html.Li(f"{row.label} ({row.value})") for row in protective.itertuples()
    #     ], className="mb-0"))

    # return summary


initial_shap_figure = make_shap_figure(model_rf, build_patient_dataframe(default_patient_input),)
initial_selected_gam_features = gam_features.copy()
initial_selected_population_features = POPULATION_CONTEXT_FEATURES.copy()
initial_gam_figure = make_gam_figure(build_gam_dataframe(default_patient_input), initial_selected_gam_features)
initial_rf_probability = float(model_rf.predict_proba(build_patient_dataframe(default_patient_input))[0, 1])
initial_gam_probability = predict_gam_risk(build_gam_dataframe(default_patient_input))
initial_clinical_summary = build_clinical_summary(model_rf,build_patient_dataframe(default_patient_input),initial_rf_probability,initial_gam_probability)
initial_population_summary, initial_similar_patients, initial_population_figure = build_population_context(
    default_patient_input,
    initial_selected_population_features,
)
initial_population_rules_df = build_plot_rules(filter_rules_by_outcome(RULES_DF, "all"))
initial_rule_matrix_figure = make_rule_matrix_figure(initial_population_rules_df, top_n=25)
initial_rule_detail = dbc.Alert("Click a rule column in the matrix to inspect that rule.", color="light", className="mb-0")
rule_filter_options = get_rule_filter_options(initial_population_rules_df)

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
    dcc.Store(id="active-view", data="patient"),
    
    # ---- Header Card ----

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
                [
                    dbc.Row(
                        [
                            dbc.Col(
                            html.Button(
                                "Patient View",
                                id="patient-view-button",
                                type="button",
                                className="btn btn-info w-100",
                                ),
                                width=6,
                            ),
                            dbc.Col(
                            html.Button(
                                "Population View",
                                id="population-view-button",
                                type="button",
                                className="btn btn-outline-info w-100",
                            ),
                                width=6,
                            ),
                        ],
                        className="g-3",
                    ),
                    html.Div(id="predicted-risk-value", style={"display": "none"}),
                    html.Div(id="risk-category-value", style={"display": "none"}),
                    html.Div(id="rf-gam-agreement", style={"display": "none"}),
                    html.Div(id="model-metrics-value", style={"display": "none"}),
                ]
            ),
        ],
        className="mb-4 shadow-sm",
    ),

    
    # ---- SHAP card -----
    html.Div(
        [
            dbc.Card([
                dbc.CardHeader(html.H2("Risk Predictor", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    dbc.Row([
                        # Patient input panel
                        build_patient_input_panel(data, default_patient_input),
                        dbc.Col([
                            # SHAP figure
                            html.H4("SHAP Explanation", className="mb-4"),
                            dcc.Graph(id="shap-patient-graph", figure=initial_shap_figure, config={"displayModeBar": False}),
                        ], width=6),
                        dbc.Col([
                            # Clinical summary
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
                            html.H4("Population Distribution", className="mb-4"),
                            dbc.Label("Continuous variables", className="fw-semibold"),
                            dcc.Checklist(
                                id="input-population-features",
                                options=[
                                    {"label": label_map.get(feature, feature.title()), "value": feature}
                                    for feature in POPULATION_CONTEXT_FEATURES
                                ],
                                value=initial_selected_population_features,
                                inline=True,
                                inputStyle={"marginRight": "0.35rem", "marginLeft": "0.5rem"},
                                labelStyle={"marginRight": "1rem"},
                                className="mb-3",
                            ),
                            dcc.Graph(
                                id="population-distribution-graph",
                                figure=initial_population_figure,
                                config={"displayModeBar": False},
                            ),
                        ], width=8),
                    ], className="g-4"),
                ])
            ], className='mb-4'),
        ],
        id="patient-view-content",
    ),
    
    # ---- Population View page -----
    html.Div(
        [
            dbc.Card([
                dbc.CardHeader(html.H2("Population View", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    html.H4("Rule Explanation", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Association Rule Matrix", className="mb-3"),
                            dcc.Graph(
                                id="population-rule-matrix-graph",
                                figure=initial_rule_matrix_figure,
                                config={"displayModeBar": False},
                            ),
                        ], width=8),
                        dbc.Col([
                            dbc.Alert(
                                [
                                    dbc.Label("Filter by attribute values", className="fw-semibold"),
                                    dcc.Dropdown(
                                        id="population-rule-filter",
                                        options=rule_filter_options,
                                        value=[],
                                        multi=True,
                                        placeholder="Select attribute values to filter rules",
                                        className="mb-3",
                                    ),
                                    dbc.Label("Outcome Scope", className="fw-semibold"),
                                    dcc.Dropdown(
                                        id="population-rule-outcome",
                                        options=[
                                            {"label": "All strong rules", "value": "all"},
                                            {"label": "Heart disease", "value": "target=1"},
                                            {"label": "No heart disease", "value": "target=0"},
                                        ],
                                        value="all",
                                        clearable=False,
                                        className="mb-3",
                                    ),
                                    dbc.Label("Row Order", className="fw-semibold"),
                                    dcc.Dropdown(
                                        id="population-rule-row-sort",
                                        options=[
                                            {"label": "Frequency", "value": "frequency"},
                                            {"label": "Alphabetical", "value": "alphabetical"},
                                            {"label": "Attribute Group", "value": "feature"},
                                        ],
                                        value="feature",
                                        clearable=False,
                                        className="mb-3",
                                    ),
                                    dbc.Label("Column Order", className="fw-semibold"),
                                    dcc.Dropdown(
                                        id="population-rule-col-sort",
                                        options=[
                                            {"label": "Lift", "value": "lift"},
                                            {"label": "Confidence", "value": "confidence"},
                                            {"label": "Support", "value": "support"},
                                            {"label": "Rule Length", "value": "rule_len"},
                                        ],
                                        value="lift",
                                        clearable=False,
                                        className="mb-3",
                                    ),
                                    dbc.Label("Row Mode", className="fw-semibold"),
                                    dcc.Checklist(
                                        id="population-rule-collapse",
                                        options=[{"label": "Collapse to attribute groups", "value": "collapse"}],
                                        value=[],
                                        inputStyle={"marginRight": "0.35rem"},
                                        className="mb-3",
                                    ),
                                    dbc.Label("Number of Rules", className="fw-semibold"),
                                    dcc.Slider(
                                        id="population-rule-count",
                                        min=10,
                                        max=40,
                                        step=5,
                                        value=25,
                                        marks={10: "10", 20: "20", 30: "30", 40: "40"},
                                        className="mb-0",
                                    ),
                                ],
                                color="primary",
                                className="mb-4",
                            ),
                            html.H5("Selected Rule", className="mb-3"),
                            html.Div(initial_rule_detail, id="population-rule-detail"),
                        ], width=4),
                    ], className="g-4 align-items-start"),
                ])
            ], className='mb-4'),
        ],
        id="population-view-content",
        style={"display": "none"},
    ),
    
    # ---- GAM card -----
    html.Div(
        [
            dbc.Card([
                dbc.CardHeader(html.H2("Feature Effects", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    html.H4("GAM Component", className="mb-4"),
                    dbc.Label("Visible plots", className="fw-semibold"),
                    dcc.Checklist(
                        id="input-gam-features",
                        options=[
                            {"label": label_map.get(feature, feature.title()), "value": feature}
                            for feature in gam_features
                        ],
                        value=initial_selected_gam_features,
                        inline=True,
                        inputStyle={"marginRight": "0.35rem", "marginLeft": "0.5rem"},
                        labelStyle={"marginRight": "1rem"},
                        className="mb-3",
                    ),
                    dcc.Graph(id="gam-patient-graph", figure=initial_gam_figure, config={"displayModeBar": False}),
                ])
            ], className='mb-4'),
        ],
        id="patient-view-gam-content",
    ),



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


@app.callback(
    Output("active-view", "data"),
    Output("patient-view-button", "className"),
    Output("population-view-button", "className"),
    Output("patient-view-content", "style"),
    Output("patient-view-gam-content", "style"),
    Output("population-view-content", "style"),
    Input("patient-view-button", "n_clicks"),
    Input("population-view-button", "n_clicks"),
    prevent_initial_call=False,
)
def update_active_view(patient_clicks, population_clicks):
    triggered = ctx.triggered_id
    active_view = "patient" if triggered != "population-view-button" else "population"

    patient_button_class = "btn btn-info w-100" if active_view == "patient" else "btn btn-outline-info w-100"
    population_button_class = "btn btn-info w-100" if active_view == "population" else "btn btn-outline-info w-100"
    patient_style = {"display": "block"} if active_view == "patient" else {"display": "none"}
    population_style = {"display": "block"} if active_view == "population" else {"display": "none"}

    return (
        active_view,
        patient_button_class,
        population_button_class,
        patient_style,
        patient_style,
        population_style,
    )


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
    Input("input-gam-features", "value"),
    Input("input-population-features", "value"),
)
def update_shap_prediction(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    selected_gam_features,
    selected_population_features,
):
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
    population_summary, similar_patients, population_figure = build_population_context(
        patient_input,
        selected_population_features,
    )
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
        make_gam_figure(gam_df, selected_gam_features),
    )


@app.callback(
    Output("population-rule-filter", "options"),
    Output("population-rule-matrix-graph", "figure"),
    Input("population-rule-filter", "value"),
    Input("population-rule-outcome", "value"),
    Input("population-rule-row-sort", "value"),
    Input("population-rule-col-sort", "value"),
    Input("population-rule-collapse", "value"),
    Input("population-rule-count", "value"),
)
def update_population_rules_table(selected_rule_filters, outcome_mode, row_sort, col_sort, collapse_values, rule_count):
    outcome_rules = build_plot_rules(filter_rules_by_outcome(RULES_DF, outcome_mode))
    filtered_rules = filter_rules_by_antecedents(outcome_rules, selected_rule_filters)
    return (
        get_rule_filter_options(outcome_rules),
        make_rule_matrix_figure(
            filtered_rules,
            top_n=rule_count,
            row_sort=row_sort,
            col_sort=col_sort,
            collapse_rows="collapse" in (collapse_values or []),
        ),
    )


@app.callback(
    Output("population-rule-detail", "children"),
    Input("population-rule-matrix-graph", "clickData"),
    Input("population-rule-filter", "value"),
    Input("population-rule-outcome", "value"),
    Input("population-rule-col-sort", "value"),
    Input("population-rule-count", "value"),
)
def update_population_rule_detail(click_data, selected_rule_filters, outcome_mode, col_sort, rule_count):
    if not click_data or not click_data.get("points"):
        return dbc.Alert("Click a rule column in the matrix to inspect that rule.", color="light", className="mb-0")

    point = click_data["points"][0]
    customdata = point.get("customdata") or []
    if not customdata or customdata[0] != "rule":
        return dash.no_update

    filtered_rules = filter_rules_by_antecedents(
        build_plot_rules(filter_rules_by_outcome(RULES_DF, outcome_mode)),
        selected_rule_filters,
    )
    matrix_rules = get_matrix_rules(filtered_rules, top_n=rule_count, col_sort=col_sort)
    rule_index = int(customdata[1])
    if rule_index < 0 or rule_index >= len(matrix_rules):
        return dbc.Alert("Rule not available for the current selection.", color="light", className="mb-0")
    return build_rule_detail_card(matrix_rules.iloc[rule_index], label=f"Rule {rule_index + 1}")


@app.callback(
    Output("population-rule-filter", "value"),
    Input("population-rule-matrix-graph", "clickData"),
    State("population-rule-filter", "value"),
    prevent_initial_call=True,
)
def update_population_rule_filter_from_matrix(click_data, current_filters):
    if not click_data or not click_data.get("points"):
        return dash.no_update

    point = click_data["points"][0]
    customdata = point.get("customdata") or []
    if not customdata or customdata[0] != "item":
        return dash.no_update

    item_value = customdata[1]
    if "=" not in str(item_value):
        return dash.no_update

    updated_filters = list(current_filters or [])
    if item_value in updated_filters:
        updated_filters.remove(item_value)
    else:
        updated_filters.append(item_value)
    return updated_filters


# ======== Run App ========
if __name__ == "__main__":
    app.run(debug=True)
