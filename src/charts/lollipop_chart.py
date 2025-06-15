import sys
import os
import json
#from utils import lol_stats, preprocess
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from dash import html, dcc, Input, Output,callback
import numpy as np
from plotly.colors import sample_colorscale


SRC_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = SRC_DIR / "assets" / "data"

DATA_PATH    = DATA_DIR / "2024_LoL_esports_match_data_from_OraclesElixir.csv"
IMG_MAP_PATH = DATA_DIR / "champion_images.json"

with IMG_MAP_PATH.open() as f:
    IMG_MAP = json.load(f)

def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    for col in ['position', 'champion', 'result', 'patch']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df['win'] = df['result'].map({1: 1, 0: 0})
    df['pick'] = 1
    return df

def preprocess_lollipop(df: pd.DataFrame)->pd.DataFrame:
    """
    Preprocess the data for the lollipop chart.

    args:
        - df: the dataframe to process
        - top_k: The number of top pair duos to considere and group others pairs to other category
    return:
        processed dataframe
    """
    # Filter for bot and support roles
    bot_sup_df = df[df['position'].isin(['bot', 'sup'])]

    # Group by game and team to find pairings
    pairings = (
        bot_sup_df.groupby(['gameid', 'teamname'])
        .apply(lambda x: tuple(sorted(x['champion'])))
        .reset_index(name='pair')
    )

    # Count frequency of each unique pairing
    pair_counts = pairings['pair'].value_counts().reset_index()
    pair_counts.columns = ['pair', 'count']
    return pair_counts

def extract_images(pair):
    if isinstance(pair, tuple) and len(pair) == 2:
        return IMG_MAP.get(pair[0], IMG_MAP['Others']), IMG_MAP.get(pair[1], IMG_MAP['Others'])
    else:
        return IMG_MAP['Others'], IMG_MAP['Others']



def select_top_k(pair_counts:pd.DataFrame, top_k=5)->pd.DataFrame:
    # Get top k pairings for the pie chart
    top_pairs = pair_counts.head(top_k).copy()
    #other_count = pair_counts['count'].iloc[top_k:].sum()

    #others_row = pd.DataFrame([{'pair': 'Others', 'count': other_count}])
    #top_pairs = pd.concat([others_row, top_pairs], ignore_index=True)
    top_pairs.sort_values('count',ascending=False,inplace=True)
    top_pairs[['adc_img', 'sup_img']] = top_pairs['pair'].apply(lambda x: pd.Series(extract_images(x)))

    return top_pairs


df = load_data()

def create_lollipop(top_k = 5)->go.Figure:
    pair_counts = preprocess_lollipop(df)

    top_pairs = select_top_k(pair_counts, top_k)
    top_pairs.loc[:,'pair'] = top_pairs['pair'].apply(
        lambda t: f"{t[0]} & {t[1]}" if isinstance(t, tuple) else str(t)
    )
    counts = top_pairs['count']
    norm_counts = (counts - counts.min()) / (counts.max() - counts.min())
    colors = sample_colorscale("teal", norm_counts.tolist())
    fig = go.Figure()

    for i, pair_count in top_pairs.iterrows():

        color = colors[i]
        
        fig.add_trace(go.Scatter(
            x = [pair_count["pair"], pair_count['pair']],
            y=[0, pair_count['count']],
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo="skip"

        ))
        fig.add_trace(go.Scatter(
            x=[pair_count["pair"], pair_count["pair"]],
            y=[pair_count['count']],
            mode="markers",
            marker=dict(
                color=color,
                size=20,
                line=dict(color='black', width=1)  # Optional outline
            ),
            showlegend=False,
            hovertemplate=(
                "<b>Team Members</b>: %{x}<br>"
                "<b>Number of Games</b>: %{y}<extra></extra>"
            )
        ))
    images = []
    for _, row in top_pairs.iterrows():
        y_val = row['count']
        x_val = row['pair']
        
        # Offset each image a bit so they appear side by side
        images.append(dict(
            source=row['adc_img'],
            x=x_val,
            y=y_val+52,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="bottom",
            sizex=200,
            sizey=200,
            sizing="contain",
            layer="above"
        ))
        images.append(dict(
            source=row['sup_img'],
            x=x_val,
            y=y_val+210,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="bottom",
            sizex=200,
            sizey=200,
            sizing="contain",
            layer="above"
        ))

    fig.update_layout(
        images=images,
        title=f'Lollipop Chart with top {top_k} common bottom-support champions team-up',
        xaxis_title='Champion duos names',
        template="plotly_dark",
        plot_bgcolor="#272822",
        paper_bgcolor="#272822",
        font=dict(size=14),
        xaxis=dict(
        range=[-0.5, len(top_pairs) - 0.5],  
            tickmode='array',
            ticktext=top_pairs['pair'],
            tickangle=-45
        ),
        yaxis=dict(
        title='Number of games played together',
        range=[0, top_pairs['count'].max() + 600]
    ),
        #plot_bgcolor='white',
        xaxis_showgrid=True,
        yaxis_showgrid=True, 
        
    )
    #print(images)

    return fig

def layout():
    return html.Div([
    html.H1("League of Legends Bot Lane Pairings", className="title"),
    html.Label("Select Top K Pairs:"),
    dcc.Dropdown(
        id='top-k-dropdown',
        options=[{'label': str(k), 'value': k} for k in [3, 5, 7, 10]],
        value=5,
        clearable=False,
        style={'width': '200px','bg': "black"}
    ),
    dcc.Graph(id='lollipop-chart'),
    html.Div([
        html.H2("Description:"),
        html.P("This lollipop chart visualization illustrates the number of games played by bot and support champions paired together. These two roles are picked as they are the only two who play on the game lane at the start of a match. It highlights the number of games for each specific champion duo.")
    ])
    ],
    style={"backgroundColor": "#272822", "color": "#F8F8F2",
               "fontFamily": "Cinzel, serif", "padding": "1rem"},
    )
    

    
@callback(
    Output('lollipop-chart', 'figure'),
    Input('top-k-dropdown', 'value')
)
def _update_chart(top_k):
    
    return create_lollipop(top_k)
