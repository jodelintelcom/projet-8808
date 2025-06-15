import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from utils import lol_stats
from pathlib import Path

# ——— Load and Preprocess Data Globally ———
SRC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR / "assets" / "data"
DATA_PATH = DATA_DIR / "2024_LoL_esports_match_data_from_OraclesElixir.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

df['win'] = df['result'].map({1:1, 0:0})
teams_data = (
    df.groupby(['teamname', 'patch'])
    .agg(
        wins=('win', 'sum'),
        games=('win', 'count'),
        dragons=('dragons', 'sum'),
        barons=('barons', 'sum'),
        firstbloods=('firstblood', 'sum'),
        heralds=('heralds', 'sum'),
        void_grubs=('void_grubs', 'sum'),
        gold15=('golddiffat15', 'mean'),
        vision=('visionscore', 'mean')
    )
    .reset_index()
)
teams_data['win_rate'] = teams_data['wins'] / teams_data['games']

# ——— Shared Globals ———
metrics = [
    'dragons', 'barons', 'firstbloods', 'heralds',
    'void_grubs', 'gold15', 'vision', 'win_rate'
]
norm = teams_data.copy()
norm[metrics] = norm[metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
palette = ['#445fa5', '#a1b0d8', '#256579', '#6d7a93', '#96a0b5', '#2c2f3e']

# ——— Dash Layout ———
def layout():
    # Remove the Dash app creation - we're just returning layout components
    patches = ['All'] + sorted(norm['patch'].dropna().unique())
    teams = sorted(norm['teamname'].unique())

    # Return the layout components directly, not wrapped in a Dash app
    return html.Div(style={'backgroundColor':'#272822','color':'#F8F8F2','fontFamily':'Cinzel, serif'}, children=[
        html.H2("Team Performance Radar Chart", style={'textAlign':'center'}),
        html.Div([
            html.Label('Patch:', style={'marginRight':'10px'}),
            dcc.Dropdown(id='patch', options=[{'label':p,'value':p} for p in patches], value='All',
                        style={'width':'200px','color':'#000'}),
            html.Label('Teams:', style={'marginLeft':'30px'}),
            dcc.Dropdown(id='teams', options=[{'label':t,'value':t} for t in teams],
                        value=teams[:3], multi=True,
                        style={'width':'400px','color':'#000'})
        ], style={'display':'flex', 'justifyContent':'center', 'paddingBottom':'20px'}),
        dcc.Graph(id='radar-chart')
    ])

# ——— Callback ———
@callback(
    Output('radar-chart','figure'),
    Input('patch','value'),
    Input('teams','value')
)
def update_radar(selected_patch, selected_teams):
    df_plot = norm if selected_patch=='All' else norm[norm['patch']==selected_patch]
    categories = [
        'Dragon Control Rate','Baron Control Rate','First Blood Rate',
        'Rift Heralds','Void Grubs','Gold Lead @15 Min','Vision Score','Win Rate'
    ]
    fig = go.Figure()
    for i, team in enumerate(selected_teams):
        row = df_plot[df_plot['teamname']==team]
        if row.empty: continue
        values = row[metrics].iloc[0].tolist()
        values += [values[0]]
        theta = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill='toself',
            name=team,
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate='<b>%{theta}</b><br>%{r:.2f}<extra>' + team + '</extra>'
        ))

    fig.update_layout(
        template='plotly_dark',
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1]),
            bgcolor='#272822'
        ),
        showlegend=True,
        title=f"Team Radar Comparison — Patch: {selected_patch}",
        margin=dict(t=80, b=30)
    )
    return fig
