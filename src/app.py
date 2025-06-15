from dash import Dash, html, dcc, Output, Input, callback, ctx
import dash_bootstrap_components as dbc
from components.sidebar import layout as sidebar_layout, SIDEBAR_STYLE
from charts.role_heatmap import layout as h_layout
from charts.vision_scatter import layout as v_layout
from charts.radar_chart import layout as r_layout
from charts.scatter_chart import layout as scatter_layout

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

CONTENT_STYLE = {
    "margin-left": f"calc({SIDEBAR_STYLE['width']} + 2rem)",
    "padding": "2rem 1rem",
}

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar_layout(),
        html.Div(id="page-content", style=CONTENT_STYLE),
    ]
)

@callback(
    Output("page-content", "children"),
    Input("nav-vision",  "n_clicks"),
    Input("nav-heatmap", "n_clicks"),
    Input("nav-radar",   "n_clicks"),
    # Ajouter les paramètres pour le callback
    Input("nav-scatter", "n_clicks"),
    prevent_initial_call=True,
)

def render_chart(n_vision, n_heat, n_radar, n_scatter): # <- ajouter les paramètres
    if ctx.triggered_id == "nav-vision":
        return v_layout()
    if ctx.triggered_id == "nav-heatmap":
        return h_layout()
    if ctx.triggered_id == "nav-radar":
        return r_layout()
    if ctx.triggered_id == "nav-scatter":
        return scatter_layout()
    
    return v_layout()

if __name__ == "__main__":
    app.run_server(debug=True)