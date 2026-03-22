"""
Navigation menu components for the Heart Disease Analytics Dashboard
"""

import dash_bootstrap_components as dbc
import dash_html_components as html


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
