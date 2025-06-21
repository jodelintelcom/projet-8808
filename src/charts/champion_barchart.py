import pandas as pd
import glob
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import os

BASE_DIR = os.path.dirname(__file__)                   # …/project_root/charts
DATA_DIR = os.path.join(BASE_DIR, '..', 'assets', 'data')
dataset_path = os.path.abspath(DATA_DIR)               # …/project_root/assets/data

def concat_datasets(path):
    files = glob.glob(os.path.join(dataset_path, '*.csv'))

    all_df = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        all_df.append(df)

    concat_df = pd.concat(all_df, ignore_index=True)

    return concat_df

def get_consistent_color_map(df):
    """Génère un mapping de couleurs cohérent pour toutes les ligues présentes"""
    all_leagues = set()
    
    # Collecter toutes les ligues possibles dans le dataset
    temp_df = preprocess(df, league_filter='all')
    if not temp_df.empty:
        all_leagues.update(temp_df['league'].unique())
    
    # Créer un mapping fixe avec des couleurs Plotly
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
    league_colors = {}
    for i, league in enumerate(sorted(all_leagues)):
        league_colors[league] = colors[i % len(colors)]
    
    return league_colors


def preprocess(df, league_filter=None, champion_filter=None):
    # Supprimer les lignes sans nom de joueur
    df = df[df['playername'].notna()]
    df = df[df['teamname'].notna()]
    df = df[df['league'].notna()]
    df = df[df['champion'].notna()]

    # Standardiser les noms de positions
    df['position'] = df['position'].replace({
        'bot': 'Bot',
        'jng': 'Jungle', 
        'mid': 'Mid',
        'sup': 'Support',
        'top': 'Top'
    })
    
    if league_filter == 'major':
        df = df[df['league'].isin(['LCK', 'LEC', 'LCS'])]
    elif league_filter in ['LCK', 'LEC', 'LCS']:
        df = df[df['league'] == league_filter]

    # Filtrer par champion si spécifié
    if champion_filter and champion_filter != 'All':
        df = df[df['champion'] == champion_filter]

    filtered_df = df.groupby(['league', 'teamname', 'position', 'playername','champion'])['result'].agg(['sum', 'count']).reset_index()
    filtered_df.columns = ['league', 'teamname', 'position', 'playername', 'champion', 'wins', 'games_played']

     # Grouper par joueur et champion pour consolider les stats
    final_stats = filtered_df.groupby(['playername', 'champion']).agg({
        'wins': 'sum',
        'games_played': 'sum',
        'teamname': lambda x: ' / '.join(x.unique()),
        'league': lambda x: ' / '.join(x.unique()),
        'position': 'first'
    }).reset_index()

    # Calculer le winrate
    final_stats['winrate'] = (final_stats['wins'] / final_stats['games_played']) * 100

    # Trier par winrate décroissant et prendre le top X
    final_stats = final_stats.sort_values('winrate', ascending=False).head(10)

    return final_stats

def get_champions_list(df):
    df_clean = df[df['champion'].notna()]
    champions = sorted(df_clean['champion'].unique())
    return champions

def get_plot(df, league, champion):
    # Créer une copie pour éviter les modifications
    df_plot = df.copy()
    
    # Créer le texte des barres avec condition pour 0% winrate
    df_plot['bar_text'] = df_plot.apply(lambda row: 
        f"0% ({row['games_played']}G)" if row['winrate'] == 0.0 
        else f"{row['winrate']:.1f}%", axis=1)

    # Créer le graphique avec l'ordre forcé
    fig = px.bar(
        df_plot,
        x='winrate',
        y='playername',
        color='league',
        color_discrete_map=LEAGUE_COLOR_MAP,
        orientation='h',
        title=f"Top 10 Players for Champion '{champion}' ({league})",
        labels={'winrate': 'Winrate (%)', 'playername': 'Player Name'},
        text='bar_text',
        category_orders={'playername': df_plot['playername'].tolist()},
        custom_data=['teamname', 'league', 'position', 'champion', 'games_played']
    )
    
    # Créer les hover data dans le bon ordre (même ordre que df_plot)
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>' +
                    'Team: %{customdata[0]}<br>' +
                    'League: %{customdata[1]}<br>' +
                    'Position: %{customdata[2]}<br>' +
                    'Champion: %{customdata[3]}<br>' +
                    'Winrate: %{x:.1f}%<br>' +
                    'Games: %{customdata[4]}<extra></extra>',
        textposition='outside',
        textfont_color='white',
        textfont_size=12
    )

    # Étendre l'axe X pour donner plus d'espace aux labels
    max_winrate = df_plot['winrate'].max()
    fig.update_xaxes(range=[0, min(110, max_winrate + 10)])
    
    # Renommer la légende
    fig.update_layout(legend_title_text='League')

    return fig

def update_axes(fig):
    """Mettre à jour les axes"""
    fig.update_xaxes(title_text='Winrate (%)')
    fig.update_yaxes(title_text='Player Name')
    return fig

def make_figure(champion='Aatrox', league_filter='all', top_x=10):
    filter_df = preprocess(df, league_filter=league_filter, champion_filter=champion)
    fig = get_plot(filter_df, league_filter, champion)
    
    fig.update_layout(
        height=800, 
        width=1200, 
        dragmode=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="#2c2f3e",
        paper_bgcolor="#2c2f3e",
        legend_title='League',
        font=dict(
            family="Beaufort, sans-serif",
            size=12,
            color="#fff"
        )
    )
    fig = update_axes(fig)
    return fig

def layout():
    """Layout de la page"""
    fig = make_figure()
    
    return html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='champion-graph')
            ], style={'width': '75%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div([
                    html.Label("Select Champion:", style={'color': '#fff', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='champion-dropdown',
                        options=[{'label': champion, 'value': champion} for champion in CHAMPIONS_LIST],
                        value=CHAMPIONS_LIST[0] if CHAMPIONS_LIST else None,
                        style={'marginBottom': '20px'}
                    )
                ]),
                
                html.Div([
                    html.Label("Select League:", style={'color': '#fff', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='league-dropdown',
                        options=[
                            {'label': 'All Leagues', 'value': 'all'},
                            {'label': 'Major Leagues', 'value': 'major'},
                            {'label': 'LCK', 'value': 'LCK'},
                            {'label': 'LEC', 'value': 'LEC'},
                            {'label': 'LCS', 'value': 'LCS'}
                        ],
                        value='all'
                    )
                ])
            ], style={
                'width': '25%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'padding': '20px',
                'backgroundColor': '#2c2f3e'
            })
        ])
    ], style={'backgroundColor': '#2c2f3e'})

@callback(
    Output('champion-graph', 'figure'),
    Input('league-dropdown', 'value'),
    Input('champion-dropdown', 'value'),
)
def update_chart(league_value, champion_value):
    """Callback pour mettre à jour le graphique selon les filtres"""
    # Valeurs par défaut si None
    if not champion_value:
        champion_value = 'Aatrox'
    if not league_value:
        league_value = 'all'
    
    # Appliquer le filtre
    filtered_df = preprocess(df, league_filter=league_value, champion_filter=champion_value)
    
    if filtered_df.empty:
        # Créer un graphique vide si pas de données
        fig = px.bar(title=f'No data available for {champion_value} in selected leagues')
        fig.update_layout(
            height=800, 
            width=1000, 
            plot_bgcolor="#2c2f3e",
            paper_bgcolor="#2c2f3e",
            font=dict(color="#fff")
        )
        return fig
    # Créer le nouveau graphique
    new_fig = get_plot(filtered_df, league_value, champion_value)
    new_fig.update_layout(
        height=800, 
        width=1000, 
        dragmode=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="#2c2f3e",
        paper_bgcolor="#2c2f3e",
        legend_title='League',
        font=dict(
            family="Beaufort, sans-serif",
            size=12,
            color="#fff"
        )
    )
    new_fig = update_axes(new_fig)

    return new_fig

# Charger et préprocesser les données au démarrage
df = concat_datasets(dataset_path)
LEAGUE_COLOR_MAP = get_consistent_color_map(df)
CHAMPIONS_LIST = get_champions_list(df)
filter_df = preprocess(df)