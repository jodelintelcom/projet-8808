import pandas as pd
import glob
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output, callback
import os
import plotly.graph_objects as go


def get_consistent_color_map(df):
    """Génère un mapping de couleurs cohérent pour toutes les ligues présentes"""
    all_leagues = ['LEC', 'LCK', 'LCS']
    
    # Créer un mapping fixe avec des couleurs Plotly
    colors = px.colors.qualitative.Set1
    league_colors = {}
    for i, league in enumerate(all_leagues):
        league_colors[league] = colors[i % len(colors)]
    
    return league_colors


BASE_DIR = os.path.dirname(__file__)                   
DATA_DIR = os.path.join(BASE_DIR, '..', 'assets', 'data')
dataset_path = os.path.abspath(DATA_DIR)               

def concat_datasets(path):
    files = glob.glob(os.path.join(dataset_path, '*.csv'))

    all_df = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        all_df.append(df)

    concat_df = pd.concat(all_df, ignore_index=True)

    return concat_df


def preprocess(df, position_filter=None, league_filter=None, stat_column=None):
    """Préprocesse les données pour comparer les stats en victoire vs défaite"""
    # Supprimer les lignes sans nom de joueur
    df = df[df['playername'].notna()]
    df = df[df['teamname'].notna()]
    df = df[df['league'].notna()]

    # Standardiser les noms de positions
    df['position'] = df['position'].replace({
        'bot': 'Bot',
        'jng': 'Jungle', 
        'mid': 'Mid',
        'sup': 'Support',
        'top': 'Top'
    })
    
    # Filtrer par ligue (seulement LEC, LCK, LCS)
    if league_filter in ['LCK', 'LEC', 'LCS']:
        df = df[df['league'] == league_filter]
    
    # Filtrer par position si spécifié
    if position_filter and position_filter != 'All':
        df = df[df['position'] == position_filter]

    # Vérifier que la colonne stat existe
    if stat_column not in df.columns:
        return pd.DataFrame()
    
    # Séparer les victoires et défaites
    wins_df = df[df['result'] == 1].copy()
    losses_df = df[df['result'] == 0].copy()
    
    # Calculer les moyennes par joueur pour les victoires et défaites
    wins_stats = wins_df.groupby(['league', 'teamname', 'position', 'playername']).agg({
        stat_column: 'mean',
        'result': 'count'
    }).reset_index()
    wins_stats.columns = ['league', 'teamname', 'position', 'playername', f'{stat_column}_wins', 'wins_count']
    
    losses_stats = losses_df.groupby(['league', 'teamname', 'position', 'playername']).agg({
        stat_column: 'mean',
        'result': 'count'
    }).reset_index()
    losses_stats.columns = ['league', 'teamname', 'position', 'playername', f'{stat_column}_losses', 'losses_count']
    
    # Fusionner les données
    merged_df = pd.merge(wins_stats, losses_stats, 
                        on=['league', 'teamname', 'position', 'playername'], 
                        how='outer')
    
    # Remplir les valeurs manquantes avec 0 pour les comptes et NaN pour les stats
    merged_df['wins_count'] = merged_df['wins_count'].fillna(0)
    merged_df['losses_count'] = merged_df['losses_count'].fillna(0)
    
    # Calculer le total de games
    merged_df['total_games'] = merged_df['wins_count'] + merged_df['losses_count']
    
    # Filtrer les joueurs avec au moins 10 matchs AU TOTAL
    min_games = 10
    merged_df = merged_df[merged_df['total_games'] >= min_games]
    
    # Calculer la moyenne globale de la stat pour le tri
    merged_df['avg_stat'] = ((merged_df[f'{stat_column}_wins'].fillna(0) * merged_df['wins_count'] + 
                              merged_df[f'{stat_column}_losses'].fillna(0) * merged_df['losses_count']) / 
                             merged_df['total_games'])
    
    # Prendre les 5 meilleurs joueurs basés sur la moyenne globale
    merged_df = merged_df.sort_values('avg_stat', ascending=False).head(5)
    
    return merged_df


def get_plot(df, position_filter, stat_column, league_filter):
    if df.empty:
        return px.bar(title=f'No data available for {position_filter} in {league_filter}')
    
    # Préparer les données pour le graphique
    plot_data = []
    
    for _, row in df.iterrows():
        # Barre pour les victoires
        if not pd.isna(row[f'{stat_column}_wins']):
            plot_data.append({
                'playername': row['playername'],
                'stat_value': row[f'{stat_column}_wins'],
                'result_type': 'Wins',
                'teamname': row['teamname'],
                'league': row['league'],
                'position': row['position'],
                'games_count': int(row['wins_count'])
            })
        
        # Barre pour les défaites
        if not pd.isna(row[f'{stat_column}_losses']):
            plot_data.append({
                'playername': row['playername'],
                'stat_value': row[f'{stat_column}_losses'],
                'result_type': 'Losses',
                'teamname': row['teamname'],
                'league': row['league'],
                'position': row['position'],
                'games_count': int(row['losses_count'])
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    if plot_df.empty:
        return px.bar(title=f'No data available for {position_filter} in {league_filter}')
    
    # Créer le graphique
    fig = px.bar(
        plot_df,
        x='playername',
        y='stat_value',
        color='result_type',
        barmode='group',
        title=f'Top 5 {position_filter} Players - {stat_column.upper()} in Wins vs Losses ({league_filter})',
        color_discrete_map={'Wins': '#4CAF50', 'Losses': '#F44336'},
        hover_data={
            'teamname': True,
            'league': True,
            'position': True,
            'stat_value': ':.2f',
            'games_count': True
        }
    )
    
    return fig


def update_axes(fig):
    """Mettre à jour les axes"""
    fig.update_xaxes(title_text='Player Name')
    fig.update_yaxes(title_text='Stat Value')
    return fig


def make_figure(position='Mid', league_filter='LCK', stat_column='dpm'):
    filter_df = preprocess(df, position_filter=position, league_filter=league_filter, stat_column=stat_column)

    """Créer la figure avec le style"""
    fig = get_plot(filter_df, position_filter=position, stat_column=stat_column, league_filter=league_filter)
    fig.update_layout(
        height=800, 
        width=1000, 
        dragmode=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="#2c2f3e",
        paper_bgcolor="#2c2f3e",
        legend_title='Result Type',
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

    return html.Div(className='stats-comparison', children=[
        html.Header(children=[
            html.H1('Player Stats Comparison'),
            html.H2('Wins vs Losses Performance')
        ]),
        html.Main(className='viz-container', children=[
            html.Div(
                className='dropdown-menus',
                children=[
                    html.Div([
                        html.Label('Select Role:', style={'color': '#fff', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id='position-dropdown',
                            options=[
                                {'label': 'Top Lane', 'value': 'Top'},
                                {'label': 'Jungle', 'value': 'Jungle'},
                                {'label': 'Mid Lane', 'value': 'Mid'},
                                {'label': 'Bot Lane', 'value': 'Bot'},
                                {'label': 'Support', 'value': 'Support'}
                            ],
                            value='Mid',
                            clearable=False,
                            style={'width': '200px'}
                        )
                    ], style={'display': 'inline-block', 'margin-right': '20px'}),
                    
                    html.Div([
                        html.Label('Select League:', style={'color': '#fff', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id='league-dropdown',
                            options=[
                                {'label': 'LCK', 'value': 'LCK'},
                                {'label': 'LEC', 'value': 'LEC'},
                                {'label': 'LCS', 'value': 'LCS'}
                            ],
                            value='LCK',
                            clearable=False,
                            style={'width': '200px'}
                        )
                    ], style={'display': 'inline-block', 'margin-right': '20px'}),
                    
                    html.Div([
                        html.Label('Select Stat:', style={'color': '#fff', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id='stat-dropdown',
                            options=[
                                {'label': 'Damage Per Minute', 'value': 'dpm'},
                                {'label': 'Vision Score Per Minute', 'value': 'vspm'},
                                {'label': 'Wards Per Minute', 'value': 'wpm'},
                                {'label': 'Damage Taken Per Minute', 'value': 'damagetakenperminute'},
                                {'label': 'Earned Gold Per Minute', 'value': 'earned gpm'}
                            ],
                            value='dpm',
                            clearable=False,
                            style={'width': '250px'}
                        )
                    ], style={'display': 'inline-block'})
                ],
                style={'margin-bottom': '20px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start'}
            ),
            
            dcc.Graph(
                id='stats-comparison-graph', 
                className='graph', 
                figure=fig, 
                config=dict(
                    scrollZoom=False,
                    showTips=False,
                    showAxisDragHandles=False,
                    doubleClick=False,
                    displayModeBar=False,
                )
            ),
        ])
    ])


@callback(
    Output('stats-comparison-graph', 'figure'),
    Input('position-dropdown', 'value'),
    Input('league-dropdown', 'value'),
    Input('stat-dropdown', 'value')
)
def update_stats_comparison_chart(position_value, league_value, stat_value):
    """Callback pour mettre à jour le graphique selon les filtres"""
    
    # Appliquer les filtres
    filtered_df = preprocess(df, position_filter=position_value, league_filter=league_value, stat_column=stat_value)
    
    # Créer le nouveau graphique
    new_fig = get_plot(filtered_df, position_filter=position_value, stat_column=stat_value, league_filter=league_value)
    new_fig.update_layout(
        height=800, 
        width=1000, 
        dragmode=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="#2c2f3e",
        paper_bgcolor="#2c2f3e",
        legend_title='Result Type',
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
filter_df = preprocess(df)
