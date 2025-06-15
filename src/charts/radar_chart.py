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

# ——— Fixed Radar Chart Callback ———
@callback(
    Output('radar-chart','figure'),
    Input('patch','value'),
    Input('teams','value')
)
def update_radar(selected_patch, selected_teams):
    if selected_patch == 'All':
        # Aggregate data across all patches for each team
        df_plot = (
            teams_data.groupby('teamname')
            .agg(
                wins=('wins', 'sum'),
                games=('games', 'sum'),
                dragons=('dragons', 'sum'),
                barons=('barons', 'sum'),
                firstbloods=('firstbloods', 'sum'),
                heralds=('heralds', 'sum'),
                void_grubs=('void_grubs', 'sum'),
                gold15=('gold15', 'mean'),  # Average gold diff at 15
                vision=('vision', 'mean')   # Average vision score
            )
            .reset_index()
        )
        # Recalculate win rate from aggregated wins/games
        df_plot['win_rate'] = df_plot['wins'] / df_plot['games']
        
        # Normalize the aggregated data
        df_plot[metrics] = df_plot[metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        # Use existing logic for specific patch
        df_plot = norm[norm['patch'] == selected_patch]
    
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

# ——— Fixed GPT Summary Callback ———
@callback(
    Output('gpt-summary', 'children'),
    Input('patch', 'value'),
    Input('teams', 'value')
)
def generate_summary(patch, teams):
    if not teams:
        return "Select at least one team to generate a performance summary."

    if patch == 'All':
        # Aggregate data across all patches for each team (same logic as radar chart)
        df_patch = (
            teams_data.groupby('teamname')
            .agg(
                wins=('wins', 'sum'),
                games=('games', 'sum'),
                dragons=('dragons', 'sum'),
                barons=('barons', 'sum'),
                firstbloods=('firstbloods', 'sum'),
                heralds=('heralds', 'sum'),
                void_grubs=('void_grubs', 'sum'),
                gold15=('gold15', 'mean'),  # Average gold diff at 15
                vision=('vision', 'mean')   # Average vision score
            )
            .reset_index()
        )
        # Recalculate win rate from aggregated wins/games
        df_patch['win_rate'] = df_patch['wins'] / df_patch['games']
        
        # Normalize the aggregated data for consistency with radar chart
        df_patch[metrics] = df_patch[metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        # Use existing logic for specific patch
        df_patch = norm[norm['patch'] == patch]

    summaries = []
    for team in teams:
        row = df_patch[df_patch['teamname'] == team]
        if row.empty:
            continue
        metrics_values = row[metrics].iloc[0]
        
        # Add raw stats context for better analysis
        if patch == 'All':
            raw_row = (
                teams_data.groupby('teamname')
                .agg(
                    total_wins=('wins', 'sum'),
                    total_games=('games', 'sum'),
                    total_dragons=('dragons', 'sum'),
                    total_barons=('barons', 'sum'),
                    total_firstbloods=('firstbloods', 'sum'),
                    total_heralds=('heralds', 'sum'),
                    total_void_grubs=('void_grubs', 'sum'),
                    avg_gold15=('gold15', 'mean'),
                    avg_vision=('vision', 'mean')
                )
                .reset_index()
            )
            raw_stats = raw_row[raw_row['teamname'] == team].iloc[0] if not raw_row[raw_row['teamname'] == team].empty else None
            
            if raw_stats is not None:
                summary = f"{team} (All Patches): Win Rate: {raw_stats['total_wins']}/{raw_stats['total_games']} ({raw_stats['total_wins']/raw_stats['total_games']:.1%}), " + \
                         f"Dragons: {raw_stats['total_dragons']}, Barons: {raw_stats['total_barons']}, " + \
                         f"First Bloods: {raw_stats['total_firstbloods']}, Heralds: {raw_stats['total_heralds']}, " + \
                         f"Void Grubs: {raw_stats['total_void_grubs']}, Avg Gold@15: {raw_stats['avg_gold15']:.0f}, " + \
                         f"Avg Vision: {raw_stats['avg_vision']:.1f}. " + \
                         f"Normalized scores: " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics_values.items())
            else:
                summary = f"{team}: " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics_values.items())
        else:
            # For specific patch, get raw data too
            raw_row = teams_data[(teams_data['patch'] == patch) & (teams_data['teamname'] == team)]
            if not raw_row.empty:
                raw_stats = raw_row.iloc[0]
                summary = f"{team} (Patch {patch}): Win Rate: {raw_stats['wins']}/{raw_stats['games']} ({raw_stats['win_rate']:.1%}), " + \
                         f"Dragons: {raw_stats['dragons']}, Barons: {raw_stats['barons']}, " + \
                         f"First Bloods: {raw_stats['firstbloods']}, Heralds: {raw_stats['heralds']}, " + \
                         f"Void Grubs: {raw_stats['void_grubs']}, Gold@15: {raw_stats['gold15']:.0f}, " + \
                         f"Vision: {raw_stats['vision']:.1f}. " + \
                         f"Normalized scores: " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics_values.items())
            else:
                summary = f"{team}: " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics_values.items())
        
        summaries.append(summary)

    prompt_context = "across all patches" if patch == 'All' else f"in patch '{patch}'"
    
    prompt = f"""
                You are a League of Legends data analyst. Given both raw statistics and normalized metrics from a radar chart, provide a short story-like performance summary of these teams {prompt_context}.
                
                Data includes:
                - Raw stats: actual wins/games, objective counts, averages
                - Normalized scores: 0-1 scale for comparison between teams
                
                Team Data:
                {chr(10).join(summaries)}

                Write in an analytical but engaging tone (2-3 short paragraphs). Emphasize key strengths, trends, or weaknesses based on both raw performance and relative standings. 
                When analyzing "All patches", focus on overall tournament/season performance.
                Avoid technical jargon and focus on storytelling that highlights what makes each team unique.
                """

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful esports analyst who provides engaging performance summaries."},
                {"role": "user", "content": prompt}
            ]
        )

        html_content = response.choices[0].message.content.strip()
        if not html_content:
            return "No summary generated... "
        
        # Parse **bold** text and convert to HTML
        def parse_bold_text(text):
            import re
            # Split text by **bold** patterns
            parts = re.split(r'\*\*(.*?)\*\*', text)
            elements = []
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Regular text
                    if part:
                        elements.append(part)
                else:  # Bold text
                    if part:
                        elements.append(html.B(part))
            return elements


         # Process paragraphs and parse bold text
        paragraphs = []
        for paragraph in html_content.split("\n\n"):
            if paragraph.strip():
                parsed_elements = parse_bold_text(paragraph.strip())
                paragraphs.append(html.P(parsed_elements))
        
        return html.Div(paragraphs)
    except Exception as e:
        return f"Error generating summary: {str(e)}"