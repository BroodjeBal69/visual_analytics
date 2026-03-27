from __future__ import annotations

from dash import Dash, Input, Output, dcc, html, dash_table, ALL, State, ctx
import dash
import dash_bootstrap_components as dbc
import json
import palette as palette_module
import scatter as scatter_module
import population as population_module
import rule_mining as rule_mining_module
import shap_card as shap_card_module
import gam_card as gam_card_module

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
    make_cluster_pca_figure,
    make_cluster_profile_figure,
    make_feature_relationships_summary,
    make_population_summary_alert,
    make_population_distribution_view_figure,
)
from rule_mining import (
    build_rule_cards,
    build_rule_detail_card,
    get_patient_rule_selection,
    get_ranked_rules_for_patient,
    build_selected_rule_details,
    build_top_rules_table_data,
    filter_rules_by_antecedents,
    get_selected_rule_matching_patient_ids,
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

# Switches between color modes and applies the selected palette to all modules
def sync_palette_mode(mode: str | None) -> str:
    resolved_mode = palette_module.apply_palette_mode(mode)

    scatter_module.CLUSTER_COLOR_MAP = palette_module.CLUSTER_COLOR_MAP
    scatter_module.DISEASE_COLOR_MAP = palette_module.DISEASE_COLOR_MAP
    scatter_module.PATIENT_COLOR = palette_module.PATIENT_COLOR

    population_module.DEEMPHASIS_GREY = palette_module.DEEMPHASIS_GREY
    population_module.DISEASE_COLOR_MAP = palette_module.DISEASE_COLOR_MAP
    population_module.CLUSTER_COLOR_MAP = palette_module.CLUSTER_COLOR_MAP
    population_module.DIVERGING_COLOR_SCALE = palette_module.DIVERGING_COLOR_SCALE
    population_module.NEGATIVE_COLOR = palette_module.NEGATIVE_COLOR
    population_module.PATIENT_COLOR = palette_module.PATIENT_COLOR
    population_module.POSITIVE_COLOR = palette_module.POSITIVE_COLOR
    population_module.PROFILE_DIRECTION_COLOR_MAP = palette_module.PROFILE_DIRECTION_COLOR_MAP

    rule_mining_module.DISEASE_COLOR_MAP = palette_module.DISEASE_COLOR_MAP
    rule_mining_module.DIVERGING_COLOR_SCALE = palette_module.DIVERGING_COLOR_SCALE
    rule_mining_module.NEGATIVE_COLOR = palette_module.NEGATIVE_COLOR
    rule_mining_module.NEUTRAL_TEXT_COLOR = palette_module.NEUTRAL_TEXT_COLOR
    rule_mining_module.PATIENT_COLOR = palette_module.PATIENT_COLOR
    rule_mining_module.POSITIVE_COLOR = palette_module.POSITIVE_COLOR

    shap_card_module.NEGATIVE_COLOR = palette_module.NEGATIVE_COLOR
    shap_card_module.NEUTRAL_LINE_COLOR = palette_module.NEUTRAL_LINE_COLOR
    shap_card_module.POSITIVE_COLOR = palette_module.POSITIVE_COLOR

    gam_card_module.NEGATIVE_COLOR = palette_module.NEGATIVE_COLOR
    gam_card_module.NEUTRAL_LINE_COLOR = palette_module.NEUTRAL_LINE_COLOR
    gam_card_module.NEUTRAL_TEXT_COLOR = palette_module.NEUTRAL_TEXT_COLOR
    gam_card_module.PATIENT_COLOR = palette_module.PATIENT_COLOR
    gam_card_module.POSITIVE_COLOR = palette_module.POSITIVE_COLOR

    return resolved_mode


def color_mode_tokens(mode: str | None) -> dict[str, str]:
    return palette_module.get_palette_tokens(mode)

# ==== INIT all modules ======
sync_palette_mode(palette_module.DEFAULT_PALETTE_MODE)

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
initial_population_distribution_feature = ["age"]
initial_population_distribution_mode = "histogram"
initial_population_distribution_figure = make_population_distribution_view_figure(
    initial_population_distribution_feature,
    initial_population_distribution_mode,
)
initial_relationship_x = "age" if "age" in RELATIONSHIP_NUMERIC_FEATURES else RELATIONSHIP_NUMERIC_FEATURES[0]
initial_relationship_y = "chol" if "chol" in RELATIONSHIP_NUMERIC_FEATURES else (
    RELATIONSHIP_NUMERIC_FEATURES[1] if len(RELATIONSHIP_NUMERIC_FEATURES) > 1 else RELATIONSHIP_NUMERIC_FEATURES[0]
)
initial_relationship_color = "disease"
initial_relationship_figure = make_feature_relationships_figure(
    initial_relationship_x,
    initial_relationship_y,
    initial_relationship_color,
    show_density_contours=False,
    patient_input={},
)
initial_relationship_summary = make_feature_relationships_summary(
    initial_relationship_x,
    initial_relationship_y,
)
initial_cluster_options = get_kmeans_cluster_options(n_clusters=3)
initial_selected_cluster = initial_cluster_options[0]["value"] if initial_cluster_options else 0
initial_cluster_pca_color = "cluster"
initial_cluster_pca_figure = make_cluster_pca_figure(
    initial_selected_cluster,
    color_by=initial_cluster_pca_color,
    n_clusters=3,
    patient_input=None,
)
initial_cluster_profile_figure = make_cluster_profile_figure(initial_selected_cluster, n_clusters=3)
initial_cluster_summary = make_cluster_explanation_summary(initial_selected_cluster, n_clusters=3)

initial_cluster_options = get_kmeans_cluster_options(n_clusters=3)

# ========= App ===========

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# ========= Layout ===========
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id="active-view", data="patient"),
    dcc.Store(id="population-selected-rules", data=[]),
    dcc.Store(id="population-rule-matched-patient-ids", data=[]),
    dcc.Store(id="color-mode", data=palette_module.DEFAULT_PALETTE_MODE),
    dcc.Store(id="population-patient-overlay-enabled", data=False),
    
    # ---- Header Card ----

    dbc.Card(
        [
            dbc.CardHeader(
                html.Div(
                    [
                        dbc.Button(
                            "👁",
                            id="colorblind-mode-glyph",
                            color="light",
                            outline=True,
                            className="colorblind-mode-glyph",
                            title="Color blind mode",
                        ),
                        dbc.Button(
                            "👤",
                            id="patient-import-glyph",
                            color="light",
                            outline=True,
                            className="population-import-glyph",
                            title="Import patient into population view",
                        ),
                        dbc.Tooltip(
                            "Color blind mode: send me an alternate palette and I can wire it in.",
                            target="colorblind-mode-glyph",
                            placement="left",
                        ),
                        dbc.Tooltip(
                            "Patient import mode: toggle to project current patient into population charts and auto-select matching rules.",
                            target="patient-import-glyph",
                            placement="left",
                        ),
                        html.H1("Heart Disease Risk Dashboard", className="mb-1 p-3 bg-info-subtle text-info-emphasis text-center fw-bold"),
                        html.Div(
                            "For patient-specific risk prediction and model insights",
                            className="card-title, text-muted text-center text-lighter"
                        ),
                    ],
                    className="dashboard-header-shell",
                ),
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
                                row_selectable="single",
                                selected_rows=[],
                                style_table={"overflowX": "auto"},
                                style_cell={"textAlign": "left", "fontSize": "0.9rem", "padding": "0.5rem"},
                                style_header={"fontWeight": "600"},
                                style_data_conditional=[
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "rgba(13, 110, 253, 0.12)",
                                        "border": "1px solid rgba(13, 110, 253, 0.35)",
                                    }
                                ],
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
                dbc.CardHeader(html.H2("Decision making rules", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    html.H4("Rule Explanation", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            # ===== Rule matrix and controls ======
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
                                [ # ===== Interactive rule builder and filters ======
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
                                    # Filtering logic
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
                            dbc.Alert(
                                "No rule-linked patients selected.",
                                id="population-rule-match-status",
                                color="light",
                                className="mb-3",
                            ),
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
                            dbc.Label("Features", className="fw-semibold"),
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
                                multi=True,
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
                # ==== Feature Relationships Card ====
                dbc.CardHeader(html.H2("Feature Relationships", className="mb-0"), className='bg-light'),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            # Interaction componentn combining different features
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
                                ],
                                value=[],
                                inputStyle={"marginRight": "0.35rem", "marginLeft": "0.5rem"},
                                className="mb-3",
                            ),
                            dbc.Alert(
                                initial_relationship_summary,
                                id="feature-relationships-summary",
                                color="light",
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
                        ], width=9),
                    ], className="g-3"),
                ])
            ], className='mb-4'),
            dbc.Card([
                # ==== Cluster Explanation Card ====
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
                            dbc.Label("PCA color by", className="fw-semibold"),
                            dcc.Dropdown(
                                id="cluster-pca-color-by",
                                options=[
                                    {"label": "Cluster", "value": "cluster"},
                                    {"label": "Disease", "value": "disease"},
                                ],
                                value=initial_cluster_pca_color,
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
                                id="cluster-pca-graph",
                                figure=initial_cluster_pca_figure,
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

], id="dashboard-root", fluid=True)

# ========= Callbacks ===========


def register_categorical_badge_callback(field_name):
    values = sorted(data[field_name].dropna().unique().tolist())

    @app.callback(
        Output(f"input-{field_name}", "data"),
        [Output({"type": f"{field_name}-badge", "value": value}, "className") for value in values],
        [Input({"type": f"{field_name}-badge", "value": value}, "n_clicks") for value in values]
        + [
            Input("reset-patient-button", "n_clicks"),
            Input("similar-patients-table", "selected_rows"),
        ],
        State("similar-patients-table", "data"),
        prevent_initial_call=False,
    )
    def update_categorical_badges(*args):
        table_rows = args[-1] if args else None
        triggered = ctx.triggered_id
        if triggered is None or triggered == "reset-patient-button":
            selected = default_patient_input[field_name]
        elif triggered == "similar-patients-table":
            selected_rows = ctx.triggered[0].get("value") if ctx.triggered else None
            if selected_rows and table_rows:
                row_index = int(selected_rows[0])
                if 0 <= row_index < len(table_rows):
                    raw_key = f"{field_name}_raw"
                    row_data = table_rows[row_index]
                    selected = row_data.get(raw_key, row_data.get(field_name, default_patient_input[field_name]))
                else:
                    selected = default_patient_input[field_name]
            else:
                selected = default_patient_input[field_name]
        else:
            selected = triggered["value"]

        if selected not in values:
            selected_str = str(selected)
            matched = next((value for value in values if str(value) == selected_str), None)
            selected = matched if matched is not None else default_patient_input[field_name]

        classes = [
            badge_class(value == selected, "me-2 mb-2")
            for value in values
        ]
        return (selected, *classes)

    return update_categorical_badges


for categorical_field in CATEGORICAL_BADGE_LABELS:
    register_categorical_badge_callback(categorical_field)


@app.callback(
    Output("input-age", "value"),
    Output("input-trestbps", "value"),
    Output("input-chol", "value"),
    Output("input-thalach", "value"),
    Output("input-oldpeak", "value"),
    Input("reset-patient-button", "n_clicks"),
    Input("similar-patients-table", "selected_rows"),
    State("similar-patients-table", "data"),
    prevent_initial_call=True,
)
def reset_patient_numeric_inputs(_n_clicks, selected_rows, table_rows):
    triggered = ctx.triggered_id
    if triggered == "similar-patients-table" and selected_rows and table_rows:
        row_index = int(selected_rows[0])
        if 0 <= row_index < len(table_rows):
            row_data = table_rows[row_index]
            return (
                int(row_data.get("age", default_patient_input["age"])),
                float(row_data.get("trestbps_raw", default_patient_input["trestbps"])),
                float(row_data.get("chol_raw", default_patient_input["chol"])),
                float(row_data.get("thalach_raw", default_patient_input["thalach"])),
                float(row_data.get("oldpeak_raw", default_patient_input["oldpeak"])),
            )

    return (
        int(default_patient_input["age"]),
        float(default_patient_input["trestbps"]),
        float(default_patient_input["chol"]),
        float(default_patient_input["thalach"]),
        float(default_patient_input["oldpeak"]),
    )


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
    Output("color-mode", "data"),
    Input("colorblind-mode-glyph", "n_clicks"),
    State("color-mode", "data"),
    prevent_initial_call=True,
)
def toggle_color_mode(_n_clicks, current_mode):
    mode = current_mode if current_mode in {"pastel", "colorblind"} else palette_module.DEFAULT_PALETTE_MODE
    return "pastel" if mode == "colorblind" else "colorblind"


@app.callback(
    Output("population-patient-overlay-enabled", "data"),
    Input("patient-import-glyph", "n_clicks"),
    State("population-patient-overlay-enabled", "data"),
    prevent_initial_call=True,
)
def toggle_population_patient_overlay(_n_clicks, enabled):
    return not bool(enabled)


@app.callback(
    Output("patient-import-glyph", "className"),
    Output("patient-import-glyph", "title"),
    Output("patient-import-glyph", "children"),
    Input("population-patient-overlay-enabled", "data"),
)
def update_population_import_glyph(enabled):
    if enabled:
        return (
            "population-import-glyph population-import-glyph-active",
            "Patient import mode is active. Click to turn off.",
            "👤",
        )
    return (
        "population-import-glyph",
        "Patient import mode is off. Click to turn on.",
        "👤",
    )


@app.callback(
    Output("dashboard-root", "style"),
    Output("colorblind-mode-glyph", "title"),
    Output("colorblind-mode-glyph", "children"),
    Input("color-mode", "data"),
)
def apply_color_mode(color_mode):
    resolved_mode = sync_palette_mode(color_mode)
    tokens = color_mode_tokens(resolved_mode)
    mode_title = "Color blind mode active" if resolved_mode == "colorblind" else "Pastel mode active"
    next_hint = "Click to switch to pastel" if resolved_mode == "colorblind" else "Click to switch to color blind"
    return (
        {
            "--positive-color": tokens["positive"],
            "--negative-color": tokens["negative"],
            "--patient-color": tokens["patient"],
            "--deemphasis-grey": tokens["grey"],
        },
        f"{mode_title}. {next_hint}.",
        "👁",
    )


@app.callback(
    Output("similar-patients-table", "selected_rows"),
    Input("similar-patients-table", "active_cell"),
    Input("reset-patient-button", "n_clicks"),
    State("similar-patients-table", "data"),
    prevent_initial_call=True,
)
def select_similar_patient_row(active_cell, _reset_clicks, table_rows):
    if ctx.triggered_id == "reset-patient-button":
        return []
    if not active_cell or not table_rows:
        return []
    row_index = int(active_cell.get("row", -1))
    if row_index < 0 or row_index >= len(table_rows):
        return []
    return [row_index]


@app.callback(
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
    Input("color-mode", "data"),
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
    color_mode,
):
    sync_palette_mode(color_mode)
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

    summary = build_clinical_summary(patient_df, probability, gam_probability)
    population_summary, similar_patients, population_figure = build_population_context(
        patient_input,
        selected_population_features,
    )
    return (
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
    feature_labels = {"target": "Heart Disease"}
    for option in filter_options or []:
        item_value = str(option.get("value", ""))
        if "=" not in item_value:
            continue
        feature = item_value.split("=", 1)[0]
        if feature in seen:
            continue
        seen.add(feature)
        feature_options.append({"label": feature_labels.get(feature, feature.replace("_", " ").title()), "value": feature})

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
    Input("population-patient-overlay-enabled", "data"),
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
    Input("population-rule-filter", "value"),
    Input("population-rule-col-sort", "value"),
    Input("population-rule-count", "value"),
    State("population-selected-rules", "data"),
    prevent_initial_call=True,
)
def update_population_selected_rules(
    selected_data,
    click_data,
    _figure,
    patient_overlay_enabled,
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
    selected_rule_filters,
    col_sort,
    rule_count,
    current_selection,
):
    triggered = ctx.triggered_id

    if patient_overlay_enabled:
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
        return get_patient_rule_selection(
            patient_input,
            selected_rule_filters,
            col_sort,
            int(rule_count or 25),
            top_n=5,
        )

    if triggered == "population-patient-overlay-enabled":
        return []

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
    Input("color-mode", "data"),
)
def update_population_rules_table(selected_rule_filters, row_sort, col_sort, collapse_values, rule_count, color_mode):
    sync_palette_mode(color_mode)
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
    Output("population-rule-matched-patient-ids", "data"),
    Input("population-selected-rules", "data"),
    Input("population-rule-filter", "value"),
    Input("population-rule-col-sort", "value"),
    Input("population-rule-count", "value"),
)
def update_rule_matched_patients(selected_rule_indices, selected_rule_filters, col_sort, rule_count):
    return get_selected_rule_matching_patient_ids(
        selected_rule_indices,
        selected_items=selected_rule_filters,
        col_sort=col_sort,
        top_n=rule_count,
    )


@app.callback(
    Output("population-rule-match-status", "children"),
    Output("population-rule-match-status", "color"),
    Input("population-rule-matched-patient-ids", "data"),
)
def update_rule_match_status(matched_patient_ids):
    matched_count = len(matched_patient_ids or [])
    if matched_count:
        return f"Selected rules match {matched_count} patients.", "info"
    return "No rule-linked patients selected.", "light"


@app.callback(
    Output("population-distribution-view-graph", "figure"),
    Input("population-distribution-feature", "value"),
    Input("population-distribution-mode", "value"),
    Input("population-rule-matched-patient-ids", "data"),
    Input("color-mode", "data"),
    Input("population-patient-overlay-enabled", "data"),
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
def update_population_distribution_view(
    feature,
    mode,
    matched_patient_ids,
    color_mode,
    patient_overlay_enabled,
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
    sync_palette_mode(color_mode)
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
    return make_population_distribution_view_figure(
        feature,
        mode,
        patient_input=patient_input if patient_overlay_enabled else None,
        show_patient=bool(patient_overlay_enabled),
        matched_patient_ids=matched_patient_ids,
    )


@app.callback(
    Output("feature-relationships-graph", "figure"),
    Output("feature-relationships-summary", "children"),
    Input("relationship-x-feature", "value"),
    Input("relationship-y-feature", "value"),
    Input("relationship-color-by", "value"),
    Input("relationship-options", "value"),
    Input("population-rule-matched-patient-ids", "data"),
    Input("color-mode", "data"),
    Input("population-patient-overlay-enabled", "data"),
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
    x_feature,
    y_feature,
    color_by,
    options,
    matched_patient_ids,
    color_mode,
    patient_overlay_enabled,
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
    sync_palette_mode(color_mode)
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
    patient_outcome_label = None
    if patient_overlay_enabled:
        patient_probability = float(model_rf.predict_proba(build_patient_dataframe(patient_input))[0, 1])
        patient_outcome_label = "Disease" if patient_probability >= 0.5 else "No disease"

    option_values = options or []
    figure = make_feature_relationships_figure(
        x_feature,
        y_feature,
        color_by,
        show_density_contours="density" in option_values,
        patient_input=patient_input if patient_overlay_enabled else {},
        matched_patient_ids=matched_patient_ids,
        patient_outcome_label=patient_outcome_label,
    )
    summary = make_feature_relationships_summary(x_feature, y_feature)
    return figure, summary


@app.callback(
    Output("cluster-pca-graph", "figure"),
    Output("cluster-profile-graph", "figure"),
    Output("cluster-explanation-summary", "children"),
    Input("cluster-explanation-selected", "value"),
    Input("cluster-pca-color-by", "value"),
    Input("population-rule-matched-patient-ids", "data"),
    Input("color-mode", "data"),
    Input("population-patient-overlay-enabled", "data"),
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
def update_cluster_explanation(
    selected_cluster,
    cluster_pca_color_by,
    matched_patient_ids,
    color_mode,
    patient_overlay_enabled,
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
    sync_palette_mode(color_mode)
    cluster_value = selected_cluster if selected_cluster is not None else initial_selected_cluster
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
    return (
        make_cluster_pca_figure(
            cluster_value,
            color_by=cluster_pca_color_by,
            n_clusters=3,
            patient_input=patient_input if patient_overlay_enabled else None,
            matched_patient_ids=matched_patient_ids,
        ),
        make_cluster_profile_figure(cluster_value, n_clusters=3),
        make_cluster_explanation_summary(cluster_value, n_clusters=3),
    )


# ======== Run App ========
if __name__ == "__main__":
    app.run(debug=True)
