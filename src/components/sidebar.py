from dash import html
import dash_bootstrap_components as dbc

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "border-right": "1px solid #dee2e6"
}

import dash_bootstrap_components as dbc

def layout():
    return dbc.Nav(
        [
            # OBLIGATOIRE : id + n_clicks=0
            dbc.NavLink(
                "Vision Scatter",
                id="nav-vision",
                n_clicks=0,        # ← crée la prop n_clicks
                class_name="mb-1"
            ),
            dbc.NavLink(
                "Role Heatmap",
                id="nav-heatmap",
                n_clicks=0,
                class_name="mb-1"
            ),
            dbc.NavLink(
                 "Team Radar",
                 id="nav-radar",
                 n_clicks=0,
                 class_name="mb-1"
            ),
            dbc.NavLink(
                "Champions Scatter map",
                id="nav-scatter",
                n_clicks=0,
                class_name="mb-1"
            ),
            # Ajouter ici comme les autres
        ],
        vertical=True,
        pills=True,
        style=SIDEBAR_STYLE
    )