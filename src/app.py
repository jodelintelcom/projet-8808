from dash import Dash, html, dcc, Output, Input, callback, ctx
import dash_bootstrap_components as dbc
from components.sidebar import layout as sidebar_layout, SIDEBAR_STYLE
from charts.role_heatmap import layout as h_layout
from charts.vision_scatter import layout as v_layout
from charts.radar_chart import layout as r_layout
from charts.scatter_chart import layout as scatter_layout
from charts.lollipop_chart import layout as lollipop_layout

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

CONTENT_STYLE = {
    "margin-left": f"calc({SIDEBAR_STYLE['width']} + 2rem)",
    "padding": "2rem 1rem",
}

def serve_layout():
    return html.Div([
        dcc.Location(id="url"),
        html.Div(id="page-content")

    ])

app.layout = serve_layout

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/dashboard":
        return html.Div(
            [
                sidebar_layout(),
                html.Div(id="main-dashboard", style=CONTENT_STYLE)
            ],
            style={"display": "flex"}
        )
    else:
        return html.Div(
        [
            html.Div([
                html.H1("Welcome to our dashboard", className="landing-title"),
                html.P("Explore the visualizations by accessing the dashboard.", className="landing-subtitle"),
                dcc.Link("Enter the dashboard", href="/dashboard", className="btn btn-lg btn-primary landing-button")
            ], className="landing-content")
        ],
    className="landing-wrapper"
)



@callback(
     Output("main-dashboard", "children"),
    Input("nav-vision",  "n_clicks"),
    Input("nav-heatmap", "n_clicks"),
    Input("nav-radar",   "n_clicks"),
    Input("nav-scatter", "n_clicks"),
    Input("nav-lollipop", "n_clicks"),
    prevent_initial_call=True,
)

def render_chart(n_vision, n_heat, n_radar, n_scatter, n_lollipop): 
    if ctx.triggered_id == "nav-vision":
        return v_layout()
    if ctx.triggered_id == "nav-heatmap":
        return h_layout()
    if ctx.triggered_id == "nav-radar":
        return r_layout()
    if ctx.triggered_id == "nav-scatter":
        return scatter_layout()
    if ctx.triggered_id == "nav-lollipop":
        return lollipop_layout()    
    return v_layout()

if __name__ == "__main__":
    app.run_server(debug=True)