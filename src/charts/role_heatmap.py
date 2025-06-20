from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
from utils import lol_stats

# ---- Pré-calcul partagé ----------------------------------------------------
PATCHES, METRICS_MAP, PICK_MAP = lol_stats.precompute()

def layout():
    return html.Div(
        [
            html.H2("LoL Role Heatmap", className="text-center"),
            html.Div(
                [
                    html.Label("Patch :", className="me-2"),
                    dcc.Dropdown(
                        id="heatmap-patch",
                        options=[{"label": p, "value": p} for p in PATCHES],
                        value="All",
                        clearable=False,
                        style={"width": "200px", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.Label("Metric :", className="me-2"),
                            dcc.RadioItems(
                                id="heatmap-metric",
                                options=[
                                    {"label": "Win Rate",  "value": "win_rate"},
                                    {"label": "Pick Rate", "value": "pick_rate"},
                                ],
                                value="win_rate",
                                inline=True,
                            ),
                        ],
                        style={"display": "inline-block", "marginLeft": "50px"},
                    ),
                ],
                className="mb-3",
            ),
            dcc.Graph(
                id="role-heatmap", 
                config={"displayModeBar": False},
                style={"height": "70vh", "width": "100%", "minWidth": "800px"}  # Force minimum width
            ),
        ],
        style={
            "backgroundColor": "#272822", 
            "color": "#F8F8F2",
            "fontFamily": "Cinzel, serif", 
            "padding": "1rem",
            "height": "100vh",  # Full viewport height
            "width": "100%",    # Full width
            "boxSizing": "border-box"  # Include padding in height calculation
        },
    )

# ---- Callback (Dash 2 +: pas besoin d'app dans la signature) --------------
@callback(
    Output("role-heatmap", "figure"),
    Input("heatmap-patch", "value"),
    Input("heatmap-metric", "value"),
)
def _update_heatmap(patch, metric):
    data = METRICS_MAP if metric == "win_rate" else PICK_MAP
    m = data[patch]
    fig = go.Figure(
        go.Heatmap(
            z=m["z"], x=m["ranks"], y=m["roles"],
            text=m["txt"], customdata=m["cd"],
            hovertemplate=(
                "<b>%{y}</b><br>%{x}<br>Champion: %{text}<br>"
                + ("Win Rate" if metric == "win_rate" else "Pick Rate")
                + ": %{z:.2%}<br>Games: %{customdata:d}<extra></extra>"
            ),
            colorscale="Blues" if metric == "win_rate" else "Purples",
            colorbar=dict(
                title="Win Rate" if metric == "win_rate" else "Pick Rate",
                tickformat=".0%",
            ),
        )
    )
    # ajout miniatures
    for i, role in enumerate(m["roles"]):
        for j, url in enumerate(m["imgs"][i]):
            if url:
                fig.add_layout_image(
                    dict(
                        source=url, x=j, y=i, xref="x", yref="y",
                        sizex=1, sizey=1, xanchor="center", yanchor="middle",
                        layer="above", sizing="contain",
                    )
                )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#272822",
        paper_bgcolor="#272822",
        font=dict(size=14),
        title=f"{patch} – {'Win' if metric == 'win_rate' else 'Pick'} Rate",
        xaxis_title="Rank",
        yaxis_title="Role",
        # Additional layout improvements for full-size display
        autosize=True,
        width=None,  # Let it expand naturally
        height=None, # Let it expand naturally
        margin=dict(l=80, r=40, t=80, b=80),  # Smaller right margin
        xaxis=dict(side="bottom"),
        yaxis=dict(side="left"),
    )
    return fig