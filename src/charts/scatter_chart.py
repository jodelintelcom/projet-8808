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
#from dash_model_viewer import DashModelViewer


BASE_DIR = os.path.dirname(__file__)                 
DATA_DIR = os.path.join(BASE_DIR, '..', 'assets', 'data')

dataset_path = os.path.abspath(DATA_DIR)             

with open(Path(__file__).parent / "../assets/data/champion_images.json") as f:
    icon_map = json.load(f)     

def concat_datasets(path):
    """
    Concatenated each year's dataset into one

    Args:
        path : Path containing all datasets
    Returns :
        concatenated pandas DataFrame
    """
    files = glob.glob(os.path.join(dataset_path, '*.csv'))

    all_df = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        all_df.append(df)

    concat_df = pd.concat(all_df, ignore_index=True)

    return concat_df


def calculate_win_rate(row):
    """
    Calculate Win-Rate for each champion

    Args:
        row : Row of DataFrame
    Returns : 
        Win-Rate
    """
    return (row['total_wins'] / row['total_plays'])*100

def preprocess(df, year = None, patch = None, champion = None):
    """
    Processing of DataFrame for Champions Win-Rate scatter chart.
    Calculates total wins and total match played for each champion
    and their win-rate.

    Args:
        df : DataFrame of esports data
        year : Filtered year
        patch : Filtered patch
        champion : Filtered champion name

    Returns:
        Preprocessed Pandas DataFrame
    """

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
    """
    Get initial plot of scatter chart

    Args:
        df : PreProcessed Pandas DataFrame

    Return:
        Plotly px scatter chart figure
    """

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

def make_figure():
    """
    Get figure and update its layout

    Args:
        None
    Returns:
        Figure with custom layout
    """
    fig = get_plot(filter_df)
    fig.update_layout(
                  autosize=True,
                  margin=dict(l=40, r=40, t=10, b=40),
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

def update_axes(fig):
    """
    Update X and Y axis of scatter chart

    Args: 
        fig : Scatter chart figure

    Returns:
        fig with updated axis
    """
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



def layout():
    """
    Creates HTML layout for Web App

    Args:
        None

    Returns:
        HTML layout
    """
    fig = make_figure()

    return html.Div(className='champions', style = {'height':'100vh','overflow':'hidden', 'margin' : '0', 'display': 'flex', 'flexDirection': 'column'}, children=[
    html.Header(style = {'margin':'0'}, children=[
        html.H1('League of Legends Champions Win-Rate', style = {'color' : '#E4C678', "margin": "0 0 .4rem 0",}),
        html.P(['This interactive chart presents champions win-rate between 2023-2025.', html.Br(), \
                'By default, the data showcased in the chart is the cumulative statistics for all years. Users can ' \
                'however filterd the data by year, patch number and by champions to visualize specific statistics ' \
                'during that timeframe.', html.Br(), 'With all years combined, we see that none of the champions ' \
                'are overpowered as those who pay more matches, maintain a win-rate of 50% while champions with high win-rate ' \
                ' play less games.'
                ], 
               style = {'color' : '#E4C678', 'marginBottom': '0.1rem'})
    ]),
    html.Main(className='viz-container', style={'height' : '90vh','display' : 'flex', 'flex' :'1', "gap": "0%", 'width' : '90%', 'marginLeft' : '5%'}, children=[
        html.Div(
            className='Dropdown-menus',
            children = [
            html.Label('Select Year:', style={'color': '#E4C678', 'display' : 'block', 'marginLeft' : '9px'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label' : 'None', 'value' : 'None'}] + [{'label': str(y), 'value': y} for y in sorted(df['year'].dropna().unique())],
                placeholder='Select Year',
                clearable=True,
                className='scatter-dropdowns',
            ),
            html.Label('Select Patch:', style={'color': '#E4C678', 'display' : 'block', 'marginLeft' : '9px'}),
            dcc.Dropdown(
                id='patch-dropdown',
                options=[{'label' : 'None', 'value' : 'None'}] + [{'label': str(p), 'value': p} for p in sorted(df['patch'].dropna().unique())],
                placeholder='Select Patch',
                clearable=True,
                className='scatter-dropdowns',
            ),
            html.Label('Select Champion:', style={'color': '#E4C678', 'display' : 'block', 'marginLeft' : '9px'}),
            dcc.Dropdown(
                id='champion_name-dropdown',
                options=[{'label' : 'None', 'value' : 'None'}] + [{'label': str(p), 'value': p} for p in sorted(df['champion'].dropna().unique())],
                placeholder='Select Champion',
                clearable=True,
                className='scatter-dropdowns',
            ),
            # DashModelViewer(
            #     id="my-viewer",
            #     src="", 
            #     alt="3D Model Champion",
            #     cameraControls=True, 
            #     cameraOrbit="0deg 75deg 1.2m", 
            #     fieldOfView="35deg",             
            #     ar=True,              
            #     style={"width": "60%", "height": "60%", "margin": "auto"}
            # )
            ],
        style={
            "flex": "0 0 14rem",     
            'height': '100%',      
            'border': '2px solid #E4C678',
            'box-shadow': '0 0 10px #E4C678, 0 0 20px rgba(228,198,120,0.5)', 
            'border-radius': '15px',
            'background': '#2c2f3e',
            'margin-right': '2%',
            'margin-left' : '12px',   
        },
        )
        ,
        dcc.Graph(id='graph', className='graph',  style={"flex": "1", "minWidth": "0", 'height':'100%', 'overflow' : 'hidden'}, figure=fig, config=dict(
            scrollZoom=False,
            showTips=False,
            showAxisDragHandles=False,
            doubleClick=False,
            displayModeBar=False,
            clear_on_unhover=True,
            responsive=True,
        )),
        dcc.Tooltip(id="graph-tooltip", style={'padding': '8px', 'background' : '#343434', 'border-radius' : '15px', 'overflow': 'hidden',}),
    ])
])



@callback(
    Output('graph', 'figure'),
    Input('year-dropdown', 'value'),
    Input('patch-dropdown', 'value'),
    Input('champion_name-dropdown', 'value')
)
def update_output_div(year_value, patch_value, champion_value):

    """
    Callback functions to update scatter plot with chosen options
    from the dropdown menus

    Args:
        year_value : Selected year from the dropdown 
        patch_value : Selected patch from the dropdown
        champion_valye : Selected champion from the dropdown

    Returns:
        Filtered figure based on the dropdown values
    """

    if year_value == 'None':
        year_value = None

    if patch_value == 'None':
        patch_value = None

    if champion_value == 'None':
        champion_value = None
    
    new_filter_df = preprocess(df, year=year_value, patch=patch_value, champion = champion_value)
    new_fig = get_plot(new_filter_df)
    
    new_fig.update_layout(
                  autosize=True, 
                  margin=dict(l=40, r=40, t=10, b=40),
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
def custom_hovertemplate(hoverData):
    """
    Displays custom hovertemplate (with image) for every marker in the scatter chart

    Args:
        hoverData : marker hoverdata

    Returns: 
        HTML layout for custom hovertemplate
    """
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
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

    """
    Callback function to display a 3D model of a champion when its name
    is selected in the dropdown filter

    Args:
        champion : Name of the champion selected in the dropdown

    Returns:
        Path of 3D model file
    """

    if not champion or champion == "All":
        return "", {"display": "none"}

    model_src = f"/assets/3d_animation/{champion}.glb"            
    if model_src is None:
        return ""

    return model_src


df = concat_datasets(dataset_path)
df["icon_url"] = df["champion"].map(icon_map)
filter_df = preprocess(df)