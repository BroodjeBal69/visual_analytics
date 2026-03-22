"""
Navigation menu components for the Heart Disease Analytics Dashboard
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from data import cp_map, exang_map, fbs_map, restecg_map, sex_map, slope_map


CATEGORICAL_BADGE_LABELS = {
    "sex": sex_map,
    "cp": cp_map,
    "fbs": fbs_map,
    "restecg": restecg_map,
    "exang": exang_map,
    "slope": slope_map,
}


def slider_marks(data, column_name):
    min_value = float(data[column_name].min())
    max_value = float(data[column_name].max())

    def format_value(value):
        return str(int(value)) if float(value).is_integer() else f"{value:.1f}"

    return {
        min_value: format_value(min_value),
        max_value: format_value(max_value),
    }


def badge_class(is_selected, extra_class=""):
    color_class = "bg-primary" if is_selected else "bg-secondary"
    suffix = f" {extra_class}" if extra_class else ""
    return f"badge rounded-pill {color_class}{suffix}"


def render_categorical_badges(data, default_patient_input, field_name):
    label_map = CATEGORICAL_BADGE_LABELS[field_name]
    default_value = default_patient_input[field_name]
    values = sorted(data[field_name].dropna().unique().tolist())
    badges = []
    for value in values:
        badges.append(
            html.Span(
                label_map.get(value, str(value)),
                id={"type": f"{field_name}-badge", "value": value},
                n_clicks=0,
                className=badge_class(value == default_value, "me-2 mb-2"),
                style={"cursor": "pointer"},
            )
        )
    badges.append(dcc.Store(id=f"input-{field_name}", data=default_value))
    return html.Div(badges)


def build_patient_input_panel(data, default_patient_input):
    return dbc.Col([
        html.H4("Patient Input", className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Age"),
                dcc.Slider(
                    id="input-age",
                    min=int(data["age"].min()),
                    max=int(data["age"].max()),
                    step=1,
                    value=int(default_patient_input["age"]),
                    marks=slider_marks(data, "age"),
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], width=6),
            dbc.Col([
                dbc.Label("Sex"),
                render_categorical_badges(data, default_patient_input, "sex"),
            ], width=6),
        ], className="g-3 mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Chest Pain Type"),
                render_categorical_badges(data, default_patient_input, "cp"),
            ], width=6),
            dbc.Col([
                dbc.Label("Fasting Blood Sugar"),
                render_categorical_badges(data, default_patient_input, "fbs"),
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
                    marks=slider_marks(data, "trestbps"),
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
                    marks=slider_marks(data, "chol"),
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], width=6),
        ], className="g-3 mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Resting ECG"),
                render_categorical_badges(data, default_patient_input, "restecg"),
            ], width=6),
            dbc.Col([
                dbc.Label("Max Heart Rate"),
                dcc.Slider(
                    id="input-thalach",
                    min=float(data["thalach"].min()),
                    max=float(data["thalach"].max()),
                    step=1,
                    value=float(default_patient_input["thalach"]),
                    marks=slider_marks(data, "thalach"),
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], width=6),
        ], className="g-3 mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Exercise Angina"),
                render_categorical_badges(data, default_patient_input, "exang"),
            ], width=6),
            dbc.Col([
                dbc.Label("ST Depression"),
                dcc.Slider(
                    id="input-oldpeak",
                    min=float(data["oldpeak"].min()),
                    max=float(data["oldpeak"].max()),
                    step=0.1,
                    value=float(default_patient_input["oldpeak"]),
                    marks=slider_marks(data, "oldpeak"),
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], width=6),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("ST Slope"),
                render_categorical_badges(data, default_patient_input, "slope"),
            ], width=6),
        ], className="g-3 mb-4"),
    ], width=3)


def create_navbar():
    """Create the top navigation bar with branding"""
    navbar = dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.A(
                    dbc.Row([
                        dbc.Col(html.Img(src="assets/heart-icon.svg", height="40px"), width="auto"),
                        dbc.Col(dbc.NavbarBrand("Heart Disease Analytics", className="ms-2")),
                    ], align="center", className="g-0"),
                    href="/", style={"textDecoration": "none"}
                ), width="auto"),
                dbc.Col(width={"size": "auto", "order": "last"}),
            ], align="center", className="g-0 w-100"),
        ]),
        color="primary",
        dark=True,
        className="mb-4",
        sticky="top",
    )
    return navbar


def create_menu():
    """Create the navigation menu with tabs"""
    menu = dbc.Nav([
        dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Patient Explorer", href="/explorer", active="exact")),
        dbc.NavItem(dbc.NavLink("Risk Predictor", href="/predictor", active="exact")),
        dbc.NavItem(dbc.NavLink("Model Analysis", href="/analysis", active="exact")),
    ], pills=True, className="mb-4")
    return menu


def get_navbar_and_menu():
    """Get both navbar and menu components"""
    return create_navbar(), create_menu()
