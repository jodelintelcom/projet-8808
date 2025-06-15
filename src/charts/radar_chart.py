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

# ——— Enhanced Dash Layout ———
def layout():
    patches = ['All'] + sorted(norm['patch'].dropna().unique())
    teams = sorted(norm['teamname'].unique())
    default_teams = ['Cloud9', 'G2 Esports', 'Team Liquid'] if all(t in teams for t in ['Gen.G', 'G2 Esports', 'T1']) else teams[:3]

    return html.Div(
        style={
            'backgroundColor': '#1a1b1e',
            'fontFamily': 'Beaufort',
            'color': '#e9ecef',
            'minHeight': '100vh',
            'padding': '0',
            'margin': '0'
        }, 
        children=[
            # Header with controls
            html.Div([
                html.H1(
                    "Team Performance Dashboard", 
                    style={
                        'textAlign': 'center',
                        'margin': '0',
                        'padding': '20px 0',
                        'fontSize': '2.5rem',
                        'fontWeight': '700',
                        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent',
                        'textShadow': '0 2px 4px rgba(0,0,0,0.3)'
                    }
                ),
                
                # Control Panel
                html.Div([
                    html.Div([
                        html.Label('Patch:', style={'fontSize': '1.1rem', 'fontWeight': '500', 'marginBottom': '8px', 'color': '#a1b0d8'}),
                        dcc.Dropdown(
                            id='patch', 
                            options=[{'label': p, 'value': p} for p in patches], 
                            value='All',
                            style={
                                'backgroundColor': '#e9ecef',
                                'border': '1px solid #4a5568',
                                'borderRadius': '8px',
                                'color': '#445fa5',
                            },
                            className='custom-dropdown'
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label('Teams:', style={'fontSize': '1.1rem', 'fontWeight': '500', 'marginBottom': '8px', 'color': '#a1b0d8'}),
                        dcc.Dropdown(
                            id='teams', 
                            options=[{'label': t, 'value': t} for t in teams],
                            value=default_teams, 
                            multi=True,
                            style={
                                'backgroundColor': '#e9ecef',
                                'border': '1px solid #4a5568',
                                'borderRadius': '8px',
                                'color': '#445fa5'
                            },
                            className='custom-dropdown'
                        )
                    ], style={'flex': '2'})
                ], style={
                    'display': 'flex', 
                    'justifyContent': 'center', 
                    'alignItems': 'end',
                    'maxWidth': '800px',
                    'margin': '0 auto 30px auto',
                    'padding': '0 20px'
                })
            ], style={
                'background': 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                'borderBottom': '1px solid #2d3748',
                'marginBottom': '20px'
            }),

            # Main content area with improved layout
            html.Div([
                # Top row: AI Summary (left) and Radar Chart (right)
                html.Div([
                    # AI Summary Panel
                    html.Div([
                        html.H3("🎯 Performance Analysis", style={
                            'margin': '0 0 15px 0',
                            'fontSize': '1.4rem',
                            'fontWeight': '600',
                            'color': '#a1b0d8',
                            'borderBottom': '2px solid #667eea',
                            'paddingBottom': '8px'
                        }),
                        dcc.Loading(
                            id="loading-summary",
                            type="dot",
                            color="#667eea",
                            children=html.Div(
                                id="gpt-summary", 
                                style={
                                    'maxHeight': '400px',
                                    'overflowY': 'auto',
                                    'overflowX': 'hidden',
                                    'padding': '15px',
                                    'fontSize': '0.95rem',
                                    'lineHeight': '1.6',
                                    'backgroundColor': 'rgba(45, 55, 72, 0.5)',
                                    'borderRadius': '8px',
                                    'border': '1px solid #4a5568',
                                    'scrollbarWidth': 'thin',
                                    'scrollbarColor': '#667eea #2d3748'
                                }
                            )
                        )
                    ], style={
                        'width': '48%',
                        'backgroundColor': 'rgba(26, 32, 44, 0.8)',
                        'padding': '20px',
                        'borderRadius': '12px',
                        'border': '1px solid #2d3748',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'backdropFilter': 'blur(10px)'
                    }),
                    
                    # Radar Chart
                    html.Div([
                        dcc.Graph(
                            id='radar-chart',
                            style={'height': '500px'}
                        )
                    ], style={
                        'width': '48%',
                        'backgroundColor': 'rgba(26, 32, 44, 0.8)',
                        'borderRadius': '12px',
                        'border': '1px solid #2d3748',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'backdropFilter': 'blur(10px)',
                        'overflow': 'hidden'
                    })
                ], style={
                    'display': 'flex',
                    'justifyContent': 'space-between',
                    'marginBottom': '30px',
                    'padding': '0 20px'
                }),
                
                # Bottom row: Line Chart (full width)
                html.Div([
                    html.H3("📈 Win Rate Evolution", style={
                        'margin': '0 0 15px 0',
                        'fontSize': '1.4rem',
                        'fontWeight': '600',
                        'color': '#a1b0d8',
                        'borderBottom': '2px solid #667eea',
                        'paddingBottom': '8px'
                    }),
                    dcc.Graph(
                        id='linechart',
                        style={'height': '400px'}
                    )
                ], style={
                    'backgroundColor': 'rgba(26, 32, 44, 0.8)',
                    'padding': '20px',
                    'borderRadius': '12px',
                    'border': '1px solid #2d3748',
                    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                    'backdropFilter': 'blur(10px)',
                    'margin': '0 20px'
                })
            ])
        ]
    )

# ——— Enhanced Radar Chart Callback ———
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
        'Dragon Control','Baron Control','First Blood',
        'Rift Heralds','Void Grubs','Gold@15min','Vision Score','Win Rate'
    ]
    
    # Enhanced color palette with more vibrant colors
    enhanced_palette = [
        '#667eea', '#f093fb', '#ffecd2', '#fcb045', '#fd746c', 
        '#4ecdc4', '#96ceb4', '#a8e6cf', '#ff8a80', '#82ca9d'
    ]
    
    fig = go.Figure()
    for i, team in enumerate(selected_teams):
        row = df_plot[df_plot['teamname']==team]
        if row.empty: continue
        values = row[metrics].iloc[0].tolist()
        values += [values[0]]
        theta = categories + [categories[0]]
        
        color = enhanced_palette[i % len(enhanced_palette)]
        #team_logo_url = get_team_logo_url(team)
        team_logo_url = "https://logos-world.net/wp-content/uploads/2022/04/Cloud-9-Emblem.png"

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill='toself',
            name=team,
            line=dict(color=color, width=3),
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
            # Clean hover template without image
            hovertemplate=f'''
            <b style="color: {color}; font-size: 14px;">{team}</b><br>
            <b>%{{theta}}</b><br>
            <span style="color: {color}; font-weight: bold;">Score: %{{r:.2f}}</span>
            <extra></extra>
            ''',
            hoverlabel=dict(
                bgcolor='rgba(26, 32, 44, 0.95)',
                bordercolor=color,
                font=dict(color='#e9ecef', size=12, family='Inter')
            )
        ))

    fig.update_layout(
        template='plotly_dark',
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0,1],
                showline=False,
                gridcolor='rgba(161, 176, 216, 0.2)',
                tickcolor='#a1b0d8',
                tickfont=dict(size=10, color='#a1b0d8')
            ),
            angularaxis=dict(
                gridcolor='rgba(161, 176, 216, 0.2)',
                linecolor='rgba(161, 176, 216, 0.3)',
                tickfont=dict(size=11, color='#e9ecef')
            ),
            bgcolor='rgba(26, 32, 44, 0.0)'
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(26, 32, 44, 0.8)',
            bordercolor='rgba(102, 126, 234, 0.3)',
            borderwidth=1,
            font=dict(size=11, color='#e9ecef')
        ),
        title=dict(
            text=f"Team Performance Radar — {selected_patch}",
            font=dict(size=18, color='#e9ecef', family='Inter'),
            x=0.5,
            y=0.95
        ),
        margin=dict(t=60, b=30, l=30, r=120),
        paper_bgcolor='rgba(26, 32, 44, 0.0)',
        plot_bgcolor='rgba(26, 32, 44, 0.0)',
        font=dict(family='Inter', color='#e9ecef')
    )
    return fig

# ——— Enhanced Line Chart Callback ———
@callback(
    Output('linechart', 'figure'),
    Input('teams', 'value')
)
def update_linechart(selected_teams):
    enhanced_palette = [
        '#667eea', '#f093fb', '#ffecd2', '#fcb045', '#fd746c', 
        '#4ecdc4', '#96ceb4', '#a8e6cf', '#ff8a80', '#82ca9d'
    ]
    
    fig = go.Figure()
    for i, team in enumerate(selected_teams):
        team_data = team_win_df[team_win_df['teamname'] == team].sort_values('patch')
        color = enhanced_palette[i % len(enhanced_palette)]
        
        fig.add_trace(go.Scatter(
            x=team_data['patch'],
            y=team_data['win_rate'],
            mode='lines+markers',
            name=team,
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color, line=dict(width=2, color='white')),
            hovertemplate=f"<b>{team}</b><br>Patch: %{{x}}<br>Win Rate: %{{y:.1%}}<extra></extra>"
        ))

    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(
            title='Patch Version',
            titlefont=dict(size=14, color='#a1b0d8'),
            tickfont=dict(size=11, color='#e9ecef'),
            gridcolor='rgba(161, 176, 216, 0.1)',
            showgrid=True
        ),
        yaxis=dict(
            title='Win Rate',
            titlefont=dict(size=14, color='#a1b0d8'),
            tickfont=dict(size=11, color='#e9ecef'),
            tickformat='.0%',
            gridcolor='rgba(161, 176, 216, 0.1)',
            showgrid=True
        ),
        plot_bgcolor='rgba(26, 32, 44, 0.0)',
        paper_bgcolor='rgba(26, 32, 44, 0.0)',
        font=dict(family='Inter', color='#e9ecef'),
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=-0.15, 
            xanchor='center', 
            x=0.5,
            bgcolor='rgba(26, 32, 44, 0.8)',
            bordercolor='rgba(102, 126, 234, 0.3)',
            borderwidth=1,
            font=dict(size=11, color='#e9ecef')
        ),
        hovermode='x unified'
    )
    return fig

# ——— Enhanced GPT Summary Callback ———
@callback(
    Output('gpt-summary', 'children'),
    Input('patch', 'value'),
    Input('teams', 'value')
)
def generate_summary(patch, teams):
    if not teams:
        return html.Div([
            html.P("🎯 Select teams to generate performance analysis", 
                   style={'textAlign': 'center', 'color': '#a1b0d8', 'fontStyle': 'italic'})
        ])

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
                You are a League of Legends esports analyst. Provide an engaging, story-like performance analysis of these teams {prompt_context}.
                
                Team Data:
                {chr(10).join(summaries)}

                Write 2-3 concise paragraphs that:
                1. Highlight each team's unique strengths and playing style
                2. Compare their strategic approaches (objectives vs fighting vs scaling)
                3. Identify standout performers and potential areas for improvement
                
                Use engaging language that would excite esports fans. Include specific numbers when relevant.
                Focus on what makes each team distinctive in their gameplay approach.
                """

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert League of Legends analyst who provides engaging, insightful team performance summaries."},
                {"role": "user", "content": prompt}
            ]
        )

        html_content = response.choices[0].message.content.strip()
        if not html_content:
            return html.Div([
                html.P("⚠️ No analysis generated. Please try again.", 
                       style={'color': '#fd746c', 'textAlign': 'center'})
            ])
        
        # Parse **bold** text and convert to HTML with better styling
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
                        elements.append(html.Strong(part, style={'color': '#667eea', 'fontWeight': '600'}))
            return elements
        
        # Process paragraphs and parse bold text
        paragraphs = []
        for paragraph in html_content.split("\n\n"):
            if paragraph.strip():
                parsed_elements = parse_bold_text(paragraph.strip())
                paragraphs.append(html.P(
                    parsed_elements,
                    style={'marginBottom': '15px', 'textAlign': 'justify'}
                ))
        
        return html.Div(paragraphs)
        
    except Exception as e:
        return html.Div([
            html.P(f"⚠️ Error generating analysis: {str(e)}", 
                   style={'color': '#fd746c', 'textAlign': 'center', 'fontStyle': 'italic'})
        ])
    

