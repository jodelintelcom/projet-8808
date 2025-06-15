# charts/vision_scatter.py
from pathlib import Path
from dash import html, dcc
import pandas as pd
import plotly.graph_objects as go

# ---------- 1. Lecture & préparation des données ---------------------------
SRC_DIR  = Path(__file__).resolve().parents[1]      #  …/src
DATA_DIR = SRC_DIR / "assets" / "data"

DF = (
    pd.read_csv(DATA_DIR / "2024_LoL_esports_match_data_from_OraclesElixir.csv",
                dtype={"url": str})
      .query("datacompleteness == 'complete' and position == 'sup'")
      .dropna(subset=["wardsplaced", "wardskilled",
                      "visionscore", "gamelength", "result"])
      .assign(
          gamelength_minutes=lambda x: x.gamelength / 60,
          WPM=lambda x: x.wardsplaced / x.gamelength_minutes,
          WCPM=lambda x: x.wardskilled / x.gamelength_minutes,
          VSPM=lambda x: x.visionscore / x.gamelength_minutes,
          VisionEfficiency=lambda x: x.VSPM / (x.WPM + x.WCPM),
          win=lambda x: x.result
      )
)

# ---------- 2. Figure Plotly (aucun callback nécessaire) --------------------
def _make_figure():
    players = sorted(DF.playername.unique())

    # un trace par point (=> contrôle total de la visibilité)
    fig = go.Figure()
    for idx, row in DF.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row.WCPM], y=[row.WPM],
                mode="markers",
                marker=dict(
                    size=10,
                    color="green" if row.win == 1 else "red",
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                name=row.playername,
                hovertemplate=(
                    f"Player: {row.playername}<br>"
                    f"Champion: {row.champion}<br>"
                    f"Team: {row.teamname}<br>"
                    f"WPM: {row.WPM:.2f}<br>"
                    f"WCPM: {row.WCPM:.2f}<br>"
                    f"VSPM: {row.VSPM:.2f}<br>"
                    f"Win: {'Yes' if row.win else 'No'}<extra></extra>"
                ),
            )
        )

    # dropdown “All / par joueur”
    buttons = [
        dict(label="All Supports",
             method="update",
             args=[{"visible": [True]*len(DF)},
                   {"title": "Vision Efficiency Scatter (Supports)"}])
    ]
    for i, p in enumerate(players):
        mask = DF.playername == p
        buttons.append(
            dict(label=p, method="update",
                 args=[{"visible": mask.tolist()},
                       {"title": f"Vision Efficiency – {p}"}])
        )

    # moyennes
    avg_wpm  = DF.WPM.mean()
    avg_wcpm = DF.WCPM.mean()

    fig.update_layout(
        template="plotly_white",
        title="Vision Efficiency Scatter (Supports)",
        xaxis_title="Wards Cleared / min",
        yaxis_title="Wards Placed / min",
        showlegend=False,
        updatemenus=[dict(buttons=buttons, x=1.15, y=1, showactive=True)],
        margin=dict(t=80, r=200),
    ).add_shape(  # ligne horizontale moyenne WPM
        type="line", x0=DF.WCPM.min(), x1=DF.WCPM.max(),
        y0=avg_wpm, y1=avg_wpm,
        line=dict(color="Blue", width=2, dash="dash")
    ).add_shape(  # ligne verticale moyenne WCPM
        type="line", x0=avg_wcpm, x1=avg_wcpm,
        y0=DF.WPM.min(),  y1=DF.WPM.max(),
        line=dict(color="Orange", width=2, dash="dash")
    ).add_annotation(x=DF.WCPM.max(), y=avg_wpm,
                     text=f"Avg WPM: {avg_wpm:.2f}", showarrow=False,
                     xanchor="left", font=dict(color="Blue")
    ).add_annotation(x=avg_wcpm, y=DF.WPM.max(),
                     text=f"Avg WCPM: {avg_wcpm:.2f}", showarrow=False,
                     yanchor="top", font=dict(color="Orange")
    )

    return fig


# ---------- 3. Layout exposé à l’app principale ----------------------------
def layout():
    return html.Div(
        [
            html.H2("Vision Efficiency (Supports)", className="text-center"),
            dcc.Graph(id="vision-scatter", figure=_make_figure(),
                      config={"displayModeBar": False}),
        ],
        style={"padding": "1rem"},
    )
