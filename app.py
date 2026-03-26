from __future__ import annotations

from dash import Dash, Input, Output, dcc, html, dash_table, ALL, State, ctx
import dash
import dash_bootstrap_components as dbc
import json

from gam_card import build_gam_dataframe, predict_gam_risk, make_gam_figure
from data import load_data
from data import label_map
from menu import CATEGORICAL_BADGE_LABELS, build_patient_input_panel, badge_class
from clinical_summary import build_clinical_summary
from models.rf import model_rf, rf_accuracy, rf_auc, gam_accuracy, gam_auc
from scatter import make_feature_relationships_figure
from population import (
    POPULATION_CONTEXT_FEATURES,
    SIMILAR_PATIENT_COLUMNS,
    RELATIONSHIP_NUMERIC_FEATURES,
    build_population_context,
    get_kmeans_cluster_options,
    make_cluster_explanation_summary,
    make_cluster_overview_figure,
    make_cluster_profile_figure,
    make_feature_relationships_summary,
    make_population_summary_alert,
    make_population_distribution_view_figure,
)
from rule_mining import (
    build_rule_cards,
    build_rule_detail_card,
    get_ranked_rules_for_patient,
    build_selected_rule_details,
    build_top_rules_table_data,
    filter_rules_by_antecedents,
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
SHAP_GRAPH_HEIGHT = "58vh"
GAM_GRAPH_HEIGHT = "74vh"
POPULATION_GRAPH_HEIGHT = "40vh"
RULE_MATRIX_GRAPH_HEIGHT = "78vh"
POPULATION_VIEW_DISTRIBUTION_HEIGHT = "48vh"

def risk_color(probability):
    probability = max(0.0, min(1.0, float(probability)))
    red = int(255 * probability)
    green = int(160 * (1 - probability))
    return f"rgb({red}, {green}, 60)"

initial_shap_figure = make_shap_figure(model_rf, build_patient_dataframe(default_patient_input),)
initial_selected_gam_features = gam_features.copy()
initial_selected_population_features = POPULATION_CONTEXT_FEATURES.copy()
initial_gam_figure = make_gam_figure(build_gam_dataframe(default_patient_input), initial_selected_gam_features)
initial_rf_probability = float(model_rf.predict_proba(build_patient_dataframe(default_patient_input))[0, 1])
initial_gam_probability = predict_gam_risk(build_gam_dataframe(default_patient_input))
initial_clinical_summary = build_clinical_summary(build_patient_dataframe(default_patient_input), initial_rf_probability, initial_gam_probability)
initial_population_summary, initial_similar_patients, initial_population_figure = build_population_context(
    default_patient_input,
    initial_selected_population_features,
)
initial_population_rules_df = build_plot_rules(RULES_DF)
initial_rule_matrix_figure = make_rule_matrix_figure(initial_population_rules_df, top_n=25)
initial_rule_detail = dbc.Alert("Click a rule column in the matrix to inspect that rule.", color="light", className="mb-0")
rule_filter_options = get_rule_filter_options(initial_population_rules_df)
initial_population_distribution_feature = "age"
initial_population_distribution_mode = "histogram"
initial_population_distribution_figure = make_population_distribution_view_figure(
    initial_population_distribution_feature,
    initial_population_distribution_mode,
)
initial_relationship_x = "age" if "age" in RELATIONSHIP_NUMERIC_FEATURES else RELATIONSHIP_NUMERIC_FEATURES[0]
initial_relationship_y = "chol" if "chol" in RELATIONSHIP_NUMERIC_FEATURES else (
    RELATIONSHIP_NUMERIC_FEATURES[1] if len(RELATIONSHIP_NUMERIC_FEATURES) > 1 else RELATIONSHIP_NUMERIC_FEATURES[0]
)
initial_relationship_view_space = "raw"
initial_relationship_color = "disease"
initial_relationship_figure = make_feature_relationships_figure(
    initial_relationship_x,
    initial_relationship_y,
    initial_relationship_color,
    view_space=initial_relationship_view_space,
    show_density_contours=False,
    highlight_patient=False,
    patient_input=default_patient_input,
)
initial_relationship_summary = make_feature_relationships_summary(
    initial_relationship_x,
    initial_relationship_y,
    view_space=initial_relationship_view_space,
)
initial_cluster_options = get_kmeans_cluster_options(n_clusters=3)
initial_selected_cluster = initial_cluster_options[0]["value"] if initial_cluster_options else 0
initial_cluster_overview_figure = make_cluster_overview_figure(initial_selected_cluster, n_clusters=3)
initial_cluster_profile_figure = make_cluster_profile_figure(initial_selected_cluster, n_clusters=3)
initial_cluster_summary = make_cluster_explanation_summary(initial_selected_cluster, n_clusters=3)

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
    dcc.Store(id="population-selected-rules", data=[]),
    
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

    
    # ---- Patient view -----
    html.Div(
        [
            # ---- SHAP card -----
            dbc.Card([
                dbc.CardHeader(html.H2("Risk Predictor", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    dbc.Row([
                        # Patient input panel
                        build_patient_input_panel(data, default_patient_input),
                        dbc.Col([
                            # SHAP figure
                            html.H4("SHAP Explanation", className="mb-4"),
                            dcc.Graph(
                                id="shap-patient-graph",
                                figure=initial_shap_figure,
                                config={"displayModeBar": False, "responsive": True},
                                responsive=True,
                                style={"height": SHAP_GRAPH_HEIGHT},
                            ),
                        ], width=5),
                        dbc.Col([
                            # Clinical summary
                            dbc.Alert(
                                initial_clinical_summary,
                                id="prediction-summary",
                                color="primary",
                                className="mb-0",
                            ),
                        ], width=4),
                    ])
                ])
            ], className='mb-4'),

            # ---- Feature Effects card -----
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
                    dcc.Graph(
                        id="gam-patient-graph",
                        figure=initial_gam_figure,
                        config={"displayModeBar": False, "responsive": True},
                        responsive=True,
                        style={"height": GAM_GRAPH_HEIGHT},
                    ),
                ])
            ], className='mb-4'),
            
            # ==== Population Context Card =====
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
                                config={"displayModeBar": False, "responsive": True},
                                responsive=True,
                                style={"height": POPULATION_GRAPH_HEIGHT},
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
                                config={"displayModeBar": True, "responsive": True},
                                responsive=True,
                                style={"height": RULE_MATRIX_GRAPH_HEIGHT},
                            ),
                        ], width=8),
                        dbc.Col([
                            dbc.Alert(
                                [
                                    dbc.Label("Rule Builder", className="fw-semibold"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="rule-builder-feature",
                                                    placeholder="Choose attribute",
                                                    clearable=False,
                                                    className="mb-2",
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="rule-builder-value",
                                                    placeholder="Choose value",
                                                    clearable=False,
                                                    className="mb-2",
                                                ),
                                                width=6,
                                            ),
                                        ],
                                        className="g-2",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button(
                                                    "Add condition",
                                                    id="rule-builder-add",
                                                    color="info",
                                                    className="w-100",
                                                    n_clicks=0,
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "Clear all",
                                                    id="rule-builder-clear",
                                                    color="secondary",
                                                    className="w-100",
                                                    n_clicks=0,
                                                ),
                                                width=6,
                                            ),
                                        ],
                                        className="g-2 mb-2",
                                    ),
                                    html.Div(id="rule-builder-active", className="mb-3"),
                                    dbc.Label("Filter by attribute values", className="fw-semibold"),
                                    dcc.Dropdown(
                                        id="population-rule-filter",
                                        options=rule_filter_options,
                                        value=[],
                                        multi=True,
                                        placeholder="Select attribute values to filter rules",
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
            dbc.Card([
                # ==== Population Distribution Card ====
                dbc.CardHeader(html.H2("Population Distributions", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Feature", className="fw-semibold"),
                            dcc.Dropdown(
                                id="population-distribution-feature",
                                options=[
                                    {
                                        "label": label_map.get(col, col.title()),
                                        "value": col,
                                    }
                                    for col in data.columns
                                    if col != "target"
                                ],
                                value=initial_population_distribution_feature,
                                clearable=False,
                                className="mb-3",
                            ),
                        ], width=8),
                        dbc.Col([
                            dbc.Label("View", className="fw-semibold"),
                            dcc.RadioItems(
                                id="population-distribution-mode",
                                options=[
                                    {"label": "Histogram", "value": "histogram"},
                                    {"label": "Density", "value": "density"},
                                ],
                                value=initial_population_distribution_mode,
                                inline=True,
                                inputStyle={"marginRight": "0.35rem", "marginLeft": "0.5rem"},
                                className="mb-3",
                            ),
                        ], width=4),
                    ], className="g-3"),
                    dcc.Graph(
                        id="population-distribution-view-graph",
                        figure=initial_population_distribution_figure,
                        config={"displayModeBar": False, "responsive": True},
                        responsive=True,
                        style={"height": POPULATION_VIEW_DISTRIBUTION_HEIGHT},
                    ),
                ])
            ], className='mb-4'),
            dbc.Card([
                dbc.CardHeader(html.H2("Feature Relationships", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("View space", className="fw-semibold"),
                            dcc.Dropdown(
                                id="relationship-view-space",
                                options=[
                                    {"label": "Raw features", "value": "raw"},
                                    {"label": "PCA projection", "value": "pca"},
                                ],
                                value=initial_relationship_view_space,
                                clearable=False,
                                className="mb-3",
                            ),
                            dbc.Label("X-axis feature", className="fw-semibold"),
                            dcc.Dropdown(
                                id="relationship-x-feature",
                                options=[
                                    {"label": label_map.get(col, col.title()), "value": col}
                                    for col in RELATIONSHIP_NUMERIC_FEATURES
                                ],
                                value=initial_relationship_x,
                                clearable=False,
                                className="mb-3",
                            ),
                            dbc.Label("Y-axis feature", className="fw-semibold"),
                            dcc.Dropdown(
                                id="relationship-y-feature",
                                options=[
                                    {"label": label_map.get(col, col.title()), "value": col}
                                    for col in RELATIONSHIP_NUMERIC_FEATURES
                                ],
                                value=initial_relationship_y,
                                clearable=False,
                                className="mb-3",
                            ),
                            dbc.Label("Color by", className="fw-semibold"),
                            dcc.Dropdown(
                                id="relationship-color-by",
                                options=[
                                    {"label": "Disease", "value": "disease"},
                                    {"label": "Cluster", "value": "cluster"},
                                    {"label": "Sex", "value": "sex"},
                                    {"label": "Chest Pain Type", "value": "cp"},
                                ],
                                value=initial_relationship_color,
                                clearable=False,
                                className="mb-3",
                            ),
                            dcc.Checklist(
                                id="relationship-options",
                                options=[
                                    {"label": " Show density contours", "value": "density"},
                                    {"label": " Highlight selected patient", "value": "highlight"},
                                ],
                                value=[],
                                inputStyle={"marginRight": "0.35rem", "marginLeft": "0.5rem"},
                                className="mb-0",
                            ),
                        ], width=3),
                        dbc.Col([
                            dcc.Graph(
                                id="feature-relationships-graph",
                                figure=initial_relationship_figure,
                                config={"displayModeBar": False, "responsive": True},
                                responsive=True,
                                style={"height": POPULATION_VIEW_DISTRIBUTION_HEIGHT},
                            ),
                        ], width=7),
                        dbc.Col([
                            dbc.Alert(
                                initial_relationship_summary,
                                id="feature-relationships-summary",
                                color="light",
                                className="mb-0",
                            ),
                        ], width=2),
                    ], className="g-3"),
                ])
            ], className='mb-4'),
            dbc.Card([
                dbc.CardHeader(html.H2("Cluster Explanation (KMeans)", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Cluster", className="fw-semibold"),
                            dcc.Dropdown(
                                id="cluster-explanation-selected",
                                options=initial_cluster_options,
                                value=initial_selected_cluster,
                                clearable=False,
                                className="mb-3",
                            ),
                            dbc.Alert(
                                initial_cluster_summary,
                                id="cluster-explanation-summary",
                                color="light",
                                className="mb-0",
                            ),
                        ], width=3),
                        dbc.Col([
                            dcc.Graph(
                                id="cluster-overview-graph",
                                figure=initial_cluster_overview_figure,
                                config={"displayModeBar": False, "responsive": True},
                                responsive=True,
                                style={"height": POPULATION_VIEW_DISTRIBUTION_HEIGHT},
                            ),
                        ], width=4),
                        dbc.Col([
                            dcc.Graph(
                                id="cluster-profile-graph",
                                figure=initial_cluster_profile_figure,
                                config={"displayModeBar": False, "responsive": True},
                                responsive=True,
                                style={"height": POPULATION_VIEW_DISTRIBUTION_HEIGHT},
                            ),
                        ], width=5),
                    ], className="g-3"),
                ])
            ], className='mb-4'),
        ],
        id="population-view-content",
        style={"display": "none"},
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
        population_style,
    )


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
    summary = build_clinical_summary(patient_df, probability, gam_probability)
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
    Output("rule-builder-feature", "options"),
    Output("rule-builder-feature", "value"),
    Input("population-rule-filter", "options"),
    State("rule-builder-feature", "value"),
)
def update_rule_builder_feature_options(filter_options, current_feature):
    feature_options = []
    seen = set()
    for option in filter_options or []:
        item_value = str(option.get("value", ""))
        if "=" not in item_value:
            continue
        feature = item_value.split("=", 1)[0]
        if feature in seen:
            continue
        seen.add(feature)
        feature_options.append({"label": feature.replace("_", " ").title(), "value": feature})

    selected_feature = current_feature if current_feature in seen else (feature_options[0]["value"] if feature_options else None)
    return feature_options, selected_feature


@app.callback(
    Output("rule-builder-value", "options"),
    Output("rule-builder-value", "value"),
    Input("rule-builder-feature", "value"),
    Input("population-rule-filter", "options"),
    State("rule-builder-value", "value"),
)
def update_rule_builder_value_options(feature, filter_options, current_value):
    value_options = []
    values = []
    for option in filter_options or []:
        item_value = str(option.get("value", ""))
        if "=" not in item_value:
            continue
        item_feature, item_raw_value = item_value.split("=", 1)
        if item_feature != feature:
            continue
        value_options.append({"label": option.get("label", item_raw_value), "value": item_raw_value})
        values.append(item_raw_value)

    selected_value = current_value if current_value in values else (value_options[0]["value"] if value_options else None)
    return value_options, selected_value


@app.callback(
    Output("rule-builder-active", "children"),
    Input("population-rule-filter", "value"),
)
def render_rule_builder_active(selected_rule_filters):
    active_items = selected_rule_filters or []
    if not active_items:
        return html.Div("No active conditions", className="text-muted small")

    return html.Div(
        [
            dbc.Button(
                str(item),
                id={"type": "rule-builder-remove", "item": str(item)},
                color="light",
                outline=True,
                size="sm",
                className="me-2 mb-2",
                n_clicks=0,
            )
            for item in active_items
        ]
    )


@app.callback(
    Output("population-rule-filter", "value"),
    Input("rule-builder-add", "n_clicks"),
    Input("rule-builder-clear", "n_clicks"),
    Input({"type": "rule-builder-remove", "item": ALL}, "n_clicks"),
    Input("population-rule-matrix-graph", "clickData"),
    State("rule-builder-feature", "value"),
    State("rule-builder-value", "value"),
    State("population-rule-filter", "value"),
    prevent_initial_call=True,
)
def update_population_rule_filters(add_clicks, clear_clicks, remove_clicks, click_data, feature, value, current_filters):
    updated_filters = list(current_filters or [])
    triggered = ctx.triggered_id

    if triggered == "rule-builder-clear":
        return []

    if triggered == "rule-builder-add":
        if not feature or value is None:
            return updated_filters
        item_value = f"{feature}={value}"
        if item_value not in updated_filters:
            updated_filters.append(item_value)
        return updated_filters

    if isinstance(triggered, dict) and triggered.get("type") == "rule-builder-remove":
        item_value = str(triggered.get("item"))
        return [item for item in updated_filters if item != item_value]

    if triggered == "population-rule-matrix-graph":
        if not click_data or not click_data.get("points"):
            return dash.no_update
        point = click_data["points"][0]
        customdata = point.get("customdata") or []
        if not customdata or customdata[0] != "item":
            return dash.no_update

        item_value = customdata[1]
        if "=" not in str(item_value):
            return dash.no_update

        if item_value in updated_filters:
            updated_filters.remove(item_value)
        else:
            updated_filters.append(item_value)
        return updated_filters

    return updated_filters


@app.callback(
    Output("population-selected-rules", "data"),
    Input("population-rule-matrix-graph", "selectedData"),
    Input("population-rule-matrix-graph", "clickData"),
    Input("population-rule-matrix-graph", "figure"),
    State("population-selected-rules", "data"),
    prevent_initial_call=True,
)
def update_population_selected_rules(selected_data, click_data, _figure, current_selection):
    triggered = ctx.triggered_id

    # Clear selection whenever the matrix is rebuilt after filtering/sorting changes.
    if triggered == "population-rule-matrix-graph" and ctx.triggered[0]["prop_id"].endswith(".figure"):
        return []

    selected = list(current_selection or [])

    if triggered == "population-rule-matrix-graph" and ctx.triggered[0]["prop_id"].endswith(".selectedData"):
        if not selected_data or not selected_data.get("points"):
            return []
        rule_indices = set()
        for point in selected_data["points"]:
            customdata = point.get("customdata") or []
            if customdata and customdata[0] == "rule":
                rule_indices.add(int(customdata[1]))
        return sorted(rule_indices)

    if triggered == "population-rule-matrix-graph" and ctx.triggered[0]["prop_id"].endswith(".clickData"):
        if not click_data or not click_data.get("points"):
            return selected
        point = click_data["points"][0]
        customdata = point.get("customdata") or []
        if not customdata or customdata[0] != "rule":
            return selected
        rule_index = int(customdata[1])
        if rule_index in selected:
            selected.remove(rule_index)
        else:
            selected.append(rule_index)
        return sorted(selected)

    return selected


@app.callback(
    Output("population-rule-filter", "options"),
    Output("population-rule-matrix-graph", "figure"),
    Input("population-rule-filter", "value"),
    Input("population-rule-row-sort", "value"),
    Input("population-rule-col-sort", "value"),
    Input("population-rule-collapse", "value"),
    Input("population-rule-count", "value"),
)
def update_population_rules_table(selected_rule_filters, row_sort, col_sort, collapse_values, rule_count):
    all_rules = build_plot_rules(RULES_DF)
    filtered_rules = filter_rules_by_antecedents(all_rules, selected_rule_filters)
    return (
        get_rule_filter_options(all_rules),
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
    Input("population-selected-rules", "data"),
    Input("population-rule-filter", "value"),
    Input("population-rule-col-sort", "value"),
    Input("population-rule-count", "value"),
)
def update_population_rule_detail(selected_rule_indices, selected_rule_filters, col_sort, rule_count):
    if not selected_rule_indices:
        return dbc.Alert("Click or brush rule columns in the matrix to inspect rules.", color="light", className="mb-0")

    filtered_rules = filter_rules_by_antecedents(
        build_plot_rules(RULES_DF),
        selected_rule_filters,
    )
    matrix_rules = get_matrix_rules(filtered_rules, top_n=rule_count, col_sort=col_sort)
    return build_selected_rule_details(matrix_rules, selected_rule_indices)


@app.callback(
    Output("population-distribution-view-graph", "figure"),
    Input("population-distribution-feature", "value"),
    Input("population-distribution-mode", "value"),
)
def update_population_distribution_view(feature, mode):
    return make_population_distribution_view_figure(feature, mode)


@app.callback(
    Output("feature-relationships-graph", "figure"),
    Output("feature-relationships-summary", "children"),
    Input("relationship-view-space", "value"),
    Input("relationship-x-feature", "value"),
    Input("relationship-y-feature", "value"),
    Input("relationship-color-by", "value"),
    Input("relationship-options", "value"),
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
def update_feature_relationships(
    view_space,
    x_feature,
    y_feature,
    color_by,
    options,
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
    option_values = options or []
    figure = make_feature_relationships_figure(
        x_feature,
        y_feature,
        color_by,
        view_space=view_space,
        show_density_contours="density" in option_values,
        highlight_patient="highlight" in option_values,
        patient_input=patient_input,
    )
    summary = make_feature_relationships_summary(x_feature, y_feature, view_space=view_space)
    return figure, summary


@app.callback(
    Output("cluster-overview-graph", "figure"),
    Output("cluster-profile-graph", "figure"),
    Output("cluster-explanation-summary", "children"),
    Input("cluster-explanation-selected", "value"),
)
def update_cluster_explanation(selected_cluster):
    cluster_value = selected_cluster if selected_cluster is not None else initial_selected_cluster
    return (
        make_cluster_overview_figure(cluster_value, n_clusters=3),
        make_cluster_profile_figure(cluster_value, n_clusters=3),
        make_cluster_explanation_summary(cluster_value, n_clusters=3),
    )


# ======== Run App ========
if __name__ == "__main__":
    app.run(debug=True)
