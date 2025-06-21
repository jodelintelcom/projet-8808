import pandas as pd
import glob
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output, callback, no_update
import os
import numpy as np
from pathlib import Path
import json
import requests
from dash_model_viewer import DashModelViewer


BASE_DIR = os.path.dirname(__file__)                 
DATA_DIR = os.path.join(BASE_DIR, '..', 'assets', 'data')
dataset_path = os.path.abspath(DATA_DIR)             

with open(Path(__file__).parent / "../assets/data/champion_images.json") as f:
    icon_map = json.load(f)     

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

def preprocess(df, year = None, patch = None, champion = None):
    df = df[df['playername'].notna()]

    df = df.replace(['bot', 'jng', 'mid', 'sup', 'top'], ['Bottom', 'Jungle', 'Middle', 'Support', 'Top'])

    df = df[['year', 'patch', 'position', 'champion', 'result', 'icon_url']]

    if year is not None:
        df = df[df['year']==year]

    if patch is not None:
        df = df[df['patch']==patch]

    if champion is not None:
        df = df[df['champion']==champion]

    sum_df = df.groupby(['position', 'champion', 'icon_url'])['result'].sum().rename('total_wins')
    count_df = df.groupby(['position', 'champion', 'icon_url']).size().rename('total_plays')
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
        custom_data = ['icon_url', 'champion', 'total_plays', 'win_rate', 'position']
    )
    return fig

def get_hovertemplate():
    hover_template = (
        "<span style='display:flex;align-items:center;'>"
        "<img src='%{customdata[0]}' "
        "style='width:32px;height:32px;border-radius:4px;margin-right:8px;'>"
        "<span>"
    )

    return hover_template


def update_axes(fig):
    fig.update_xaxes(
        title_text='Match Played',
        linecolor="#E4C678", 
        tickcolor="#E4C678",
        tickfont=dict(color="#E4C678"),
        zeroline=False, 
        )
    
    
    fig.update_yaxes(
        title_text='Winning Rate (%)',
        linecolor="#E4C678", 
        tickcolor="#E4C678",
        tickfont=dict(color="#E4C678"),
        zeroline=False, 
        )

    return fig


def make_figure():
    fig = get_plot(filter_df)
    fig.update_layout(
                  #height=800, 
                  #width=1336,
                  autosize=True, 
                  dragmode=False,
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  plot_bgcolor = "#2c2f3e",
                  paper_bgcolor = "#2c2f3e",
                  legend_title = 'Champion Roles',
                  hovermode="closest",
                  hoverdistance=10,
                  font=dict(
                    family="Beaufort, sans-serif",
                    size=12,
                    color="#E4C678"
                    ),
                )
    fig.update_traces(marker=dict(size=18), hoverinfo="none", hovertemplate=None)
    fig.update_layout(hovermode="closest")
    fig = update_axes(fig)

    return fig


def layout():
    fig = make_figure()

    return html.Div(className='champions', children=[
    html.Header(children=[
        html.H1('League of Legends Champions Win-Rate', style = {'color' : '#E4C678'})
    ]),
    html.Main(className='viz-container', style={'display' : 'flex', "gap": "2%", 'height' : '90vh', 'width' : '100%'}, children=[
        html.Div(
            className='Dropdown-menus',
            children = [dcc.Dropdown(
                id='year-dropdown',
                options=[{'label' : 'All', 'value' : 'All'}] + [{'label': str(y), 'value': y} for y in sorted(df['year'].dropna().unique())],
                placeholder='Select year',
                clearable=True,
                style={'width': '160px', 'margin-top' : '15px', 'margin-left' : '12px', 'background' : '#e9ecef', 'color' : '#445fa5'}
            ),
            dcc.Dropdown(
                id='patch-dropdown',
                options=[{'label' : 'All', 'value' : 'All'}] + [{'label': str(p), 'value': p} for p in sorted(df['patch'].dropna().unique())],
                placeholder='Select patch',
                clearable=True,
                style={'width': '160px', 'margin-top' : '15px', 'margin-left' : '12px', 'background' : '#e9ecef' , 'color' : '#445fa5'}
            ),
            dcc.Dropdown(
                id='champion_name-dropdown',
                options=[{'label' : 'All', 'value' : 'All'}] + [{'label': str(p), 'value': p} for p in sorted(df['champion'].dropna().unique())],
                placeholder='Select Champion',
                clearable=True,
                style={'width': '160px', 'margin-top' : '15px', 'margin-left' : '12px', 'background' : '#e9ecef', 'color' : '#445fa5'}
            ),
            DashModelViewer(
                id="my-viewer",
                src="assets/3d_animation/varus.glb", 
                alt="3D Model Champion",
                cameraControls=True, 
                cameraOrbit="0deg 75deg 1.2m", 
                fieldOfView="35deg",             
                ar=True,              
                style={"width": "80%", "height": "80%", "margin": "auto"}
            )
            ],
        style={
            "flex": "0 0 14rem",     
            'height': '100%',      
            'border': '2px solid #E4C678',
            'box-shadow': '0 0 10px #E4C678, 0 0 20px rgba(228,198,120,0.5)', 
            'border-radius': '15px',
            'background': '#2c2f3e',
            'margin-right': '2%'   
        },
        )
        ,
        dcc.Graph(id='graph', className='graph',  style={"flex": "1", "minWidth": "0"}, figure=fig, config=dict(
            scrollZoom=False,
            showTips=False,
            showAxisDragHandles=False,
            doubleClick=False,
            displayModeBar=False,
            clear_on_unhover=True,
            responsive=True,
        )),
        dcc.Tooltip(id="graph-tooltip", style={"padding": "8px", 'background' : '#343434', 'border-radius' : '15px'}),
    ])
])



@callback(
    Output('graph', 'figure'),
    Input('year-dropdown', 'value'),
    Input('patch-dropdown', 'value'),
    Input('champion_name-dropdown', 'value')
)
def update_output_div(year_value, patch_value, champion_value):

    if year_value == 'All':
        year_value = None

    if patch_value == 'All':
        patch_value = None

    if champion_value == 'ALL':
        champion_value = None
    
    new_filter_df = preprocess(df, year=year_value, patch=patch_value, champion = champion_value)
    new_fig = get_plot(new_filter_df)
    
    new_fig.update_layout(height=800, 
                  width=1336, 
                  dragmode=False,
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  plot_bgcolor = "#2c2f3e",
                  paper_bgcolor = "#2c2f3e",
                  legend_title = 'Champion Roles',
                  hovermode="closest",
                  hoverdistance=10,
                  font=dict(
                    family="Beaufort, sans-serif",
                    size=12,
                    color="#E4C678"
                    )
                )
    
    new_fig.update_traces(marker=dict(size=18), hoverinfo="none", hovertemplate=None)
    new_fig.update_layout(hovermode="closest")
    new_fig = update_axes(new_fig)

    return new_fig



@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox     = pt["bbox"]
    icon_url = pt["customdata"][0] 

    champion_name = pt["customdata"][1]
    match_count = pt["customdata"][2]
    win_rate = pt["customdata"][3]
    position = pt['customdata'][4]

    children = [ 
        html.Div(
            children = [
                            html.Div(children = [
                html.Img(src=icon_url, style={"width":"48px","height":"48px", 'display' : 'inline-block'}), 
                html.P(champion_name, style={'color': '#EDEADE', 'display' : 'inline-block', 'padding-left' : '10px'}) 
                ],
            ),
            html.P(position, style={'color': '#E4C678', "margin-bottom" : "0"}),
            html.P(f"{match_count} games played", style={'color': '#EDEADE', "margin-bottom" : "0"}),
             html.P(f"{win_rate:.1f}% win rate", style={"color": "#EDEADE", "margin-bottom" : "0"}),
            ],

        )
    ]
    return True, bbox, children


@callback(
    Output("my-viewer", "src"),        
    Input("champion_name-dropdown", "value"),
)
def update_model(champion):

    if not champion or champion == "All":
        return "", {"display": "none"}

    model_src = f"/assets/3d_animation/{champion}.glb"            
    if model_src is None:
        return ""

    return model_src


df = concat_datasets(dataset_path)
df["icon_url"] = df["champion"].map(icon_map)
filter_df = preprocess(df)