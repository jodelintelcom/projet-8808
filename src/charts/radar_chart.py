import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, callback
from utils import lol_stats
from pathlib import Path
import os
from openai import OpenAI


# ——— Load and Preprocess Data Globally ———
SRC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR / "assets" / "data"
DATA_PATH = DATA_DIR / "2024_LoL_esports_match_data_from_OraclesElixir.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

api_key=os.environ.get("OPENAI_API_KEY"),
print("api key : ", api_key)

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

# Compute team win rates per patch
team_win_df = (
    df.groupby(['patch', 'teamname'])
    .agg(games=('result', 'count'), wins=('win', 'sum'))
    .reset_index()
)
team_win_df['win_rate'] = team_win_df['wins'] / team_win_df['games']

# ——— Dash Layout ———
def layout():
    patches = ['All'] + sorted(norm['patch'].dropna().unique())
    teams = sorted(norm['teamname'].unique())
    default_teams = ['Cloud9', 'G2 Esports', 'T1'] if all(t in teams for t in ['Gen.G', 'G2 Esports', 'T1']) else teams[:3]

    return html.Div(style={'backgroundColor':'#272822','color':'#F8F8F2','fontFamily':'Cinzel, serif'}, children=[
        html.H2("Team Performance Dashboard", style={'textAlign':'center'}),

        html.Div([
            html.Label('Patch (Radar):', style={'marginRight':'10px'}),
            dcc.Dropdown(id='patch', options=[{'label':p,'value':p} for p in patches], value='All',
                         style={'width':'200px','color':'#000'}),
            html.Label('Teams:', style={'marginLeft':'30px'}),
            dcc.Dropdown(id='teams', options=[{'label':t,'value':t} for t in teams],
                         value=default_teams, multi=True,
                         style={'width':'400px','color':'#000'})
        ], style={'display':'flex', 'justifyContent':'center', 'paddingBottom':'20px'}),

        dcc.Graph(id='radar-chart'),
        dcc.Graph(id='linechart'),
        dcc.Loading(
            id="loading-summary",
            type="default",
            children=html.Div(id="gpt-summary", style={
                'maxWidth': '1000px', 'margin': '0 auto', 'padding': '20px', 'fontSize': '18px', 'lineHeight': '1.6'})
        )
    ])

# ——— Radar Chart Callback ———
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

# ——— Line Chart Callback ———
@callback(
    Output('linechart', 'figure'),
    Input('teams', 'value')
)
def update_linechart(selected_teams):
    fig = go.Figure()
    for i, team in enumerate(selected_teams):
        team_data = team_win_df[team_win_df['teamname'] == team].sort_values('patch')
        fig.add_trace(go.Scatter(
            x=team_data['patch'],
            y=team_data['win_rate'],
            mode='lines+markers',
            name=team,
            line=dict(color=palette[i % len(palette)], width=3),
            marker=dict(size=6),
            hovertemplate=f"<b>{team}</b><br>Patch: %{{x}}<br>Win Rate: %{{y:.2%}}<extra></extra>"
        ))

    fig.update_layout(
        template='plotly_dark',
        title='Win Rate Evolution per Team by Patch',
        xaxis_title='Patch',
        yaxis_title='Win Rate',
        yaxis_tickformat='.0%',
        plot_bgcolor='#272822',
        paper_bgcolor='#272822',
        font=dict(family='Cinzel, serif', color='#F8F8F2'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
    )
    return fig

# ——— GPT Summary Callback ———
@callback(
    Output('gpt-summary', 'children'),
    Input('patch', 'value'),
    Input('teams', 'value')
)
def generate_summary(patch, teams):
    if not teams:
        return "Select at least one team to generate a performance summary."

    df_patch = norm if patch == 'All' else norm[norm['patch'] == patch]

    summaries = []
    for team in teams:
        row = df_patch[df_patch['teamname'] == team]
        if row.empty:
            continue
        metrics_values = row[metrics].iloc[0]
        summary = f"{team}: " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics_values.items())
        summaries.append(summary)

    prompt = f"""
                You are a League of Legends data analyst. Given normalized metrics from a radar chart and win rate evolution over patches, provide a short story-like performance summary of these teams in patch '{patch}'.
                Data:
                {chr(10).join(summaries)}

                Write in an analytical but engaging tone (1-3 short paragraphs). Emphasize key strengths, trends, or weaknesses.
                Avoid technical jargon and focus on storytelling.
                """

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful esports analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        html_content = response.choices[0].message.content.strip()
        if not html_content:
            return "No summary generated... "
        return html.Div([
            html.P(p) for p in html_content.split("\n\n") if p.strip()
        ])
    except Exception as e:
        return f"Error generating summary: {str(e)}"