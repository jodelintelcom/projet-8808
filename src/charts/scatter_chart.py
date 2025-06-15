import pandas as pd
import glob
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output, callback
import os
import numpy as np


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



def calculate_win_rate(row):
    return (row['total_wins'] / row['total_plays'])*100

def preprocess(df, year = None, patch = None):
    df = df[df['playername'].notna()]

    df = df.replace(['bot', 'jng', 'mid', 'sup', 'top'], ['Bottom', 'Jungle', 'Middle', 'Support', 'Top'])

    df = df[['year', 'patch', 'position', 'champion', 'result']]

    if year is not None:
        df = df[df['year']==year]

    if patch is not None:
        df = df[df['patch']==patch]

    sum_df = df.groupby(['position', 'champion'])['result'].sum().rename('total_wins')
    count_df = df.groupby(['position', 'champion']).size().rename('total_plays')
    group_df = pd.concat([sum_df, count_df], axis=1)

    new_df = group_df.reset_index()

    new_df['win_rate'] = new_df.apply(calculate_win_rate, axis=1)


    return new_df


def get_plot(df):

    fig = px.scatter(
        df,
        x = 'total_plays',
        y = 'win_rate',
        color = 'position',
        hover_name='champion',
        opacity = 0.95,
    )
    return fig


def update_axes(fig):
    fig.update_xaxes(title_text='Match Played')
    fig.update_yaxes(title_text='Winning Rate (%)')

    return fig


def make_figure():
    fig = get_plot(filter_df)
    fig.update_layout(height=600, 
                  width=1000, 
                  dragmode=False,
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  plot_bgcolor = "#2c2f3e",
                  paper_bgcolor = "#2c2f3e",
                  legend_title = 'Champion Roles',
                  font=dict(
                    family="Beaufort, sans-serif",
                    size=12,
                    color="#fff"
                    )
                )
    fig.update_traces(marker=dict(size=18))
    fig = update_axes(fig)

    return fig


def layout():
    fig = make_figure()

    return html.Div(className='champions', children=[
    html.Header(children=[
        html.H1('Champions Stats'),
        html.H2('LoL')
    ]),
    html.Main(className='viz-container', children=[
        html.Div(
            className='Dropdown-menus',
            children = [dcc.Dropdown(
                id='year-dropdown',
                options=[{'label' : 'All', 'value' : 'All'}] + [{'label': str(y), 'value': y} for y in sorted(df['year'].dropna().unique())],
                placeholder='Select year',
                clearable=True,
                style={ 'width': '200px'}
            ),
            dcc.Dropdown(
                id='patch-dropdown',
                options=[{'label' : 'All', 'value' : 'All'}] + [{'label': str(p), 'value': p} for p in sorted(df['patch'].dropna().unique())],
                placeholder='Select patch',
                clearable=True,
                style={'width': '200px'}
            )],
            style = {'display' : 'inline-block'}
        )
        ,
        dcc.Graph(id='graph', className='graph', figure=fig, config=dict(
            scrollZoom=False,
            showTips=False,
            showAxisDragHandles=False,
            doubleClick=False,
            displayModeBar=False,
        )),
    ])
])



@callback(
    Output('graph', 'figure'),
    Input('year-dropdown', 'value'),
    Input('patch-dropdown', 'value')
)
def update_output_div(year_value, patch_value):

    if year_value == 'All':
        year_value = None

    if patch_value == 'All':
        patch_value = None
    
    new_filter_df = preprocess(df, year=year_value, patch=patch_value)
    new_fig = get_plot(new_filter_df)
    
    new_fig.update_layout(height=600, 
                  width=1000, 
                  dragmode=False,
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  plot_bgcolor = "#2c2f3e",
                  paper_bgcolor = "#2c2f3e",
                  legend_title = 'Champion Roles',
                  font=dict(
                    family="Beaufort, sans-serif",
                    size=12,
                    color="#fff"
                    )
                )
    
    new_fig.update_traces(marker=dict(size=18))
    
    new_fig = update_axes(new_fig)

    return new_fig


df = concat_datasets(dataset_path)
filter_df = preprocess(df)