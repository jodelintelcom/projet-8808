from dash import html
import dash_bootstrap_components as dbc

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#1e1e2f",
    "color": "#f8f9fa",
    "border-right": "1px solid #343a40",
}

def layout():
    return html.Div([
        html.H2("Menu", className="text-white mb-4 text-center"),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-fire me-2"), "Role Heatmap"],
                    id="nav-heatmap",
                    n_clicks=0,
                    class_name="mb-2 text-white"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-bullseye me-2"), "Team Radar"],
                    id="nav-radar",
                    n_clicks=0,
                    class_name="mb-2 text-white"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-crosshairs me-2"), "Champions Scatter"],
                    id="nav-scatter",
                    n_clicks=0,
                    class_name="mb-2 text-white"
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-chart-bar me-2"), "Duos Lollipop"],
                    id="nav-lollipop",
                    n_clicks=0,
                    class_name="mb-2 text-white"
                ),
            ],
            vertical=True,
            pills=True,
            class_name="sidebar"
        )
    ], style=SIDEBAR_STYLE)