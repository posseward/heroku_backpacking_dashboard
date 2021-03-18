#!/usr/bin/env python
# coding: utf-8

# # Backpacking Dashboard
#
# Integrate Photos (with timestamps and GPS coordinates) with a GPS route test

# In[1]:


import base64
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import flask
import geopandas
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy


from dash.dependencies import Input, Output  # Load Data
from datetime import datetime
from GPSPhoto import gpsphoto
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

from functions import *

# In[4]:

#enter path to your folder of images
image_directory = 'rae_lakes_github'


# In[]
df = create_image_df(image_directory)

# In[11]:

#generate static image for display on website
list_of_images = df['ID_list'].tolist()

static_image_route = '/static/'

# In[12]:

#get geojson file
geo_filename = 'rae_lakes_2020.geojson'



# In[]
df_geo = create_geo_df(geo_filename)

# In[15]:

#find zoom and center for map
zoom, center = zoom_center(
    lons= list(df_geo.Longitude),
    lats= list(df_geo.Latitude),
    width_to_height = 4.0
)


# In[24]:

df = predict_image_route_distance(df,df_geo)

# In[26]:

#test = df['Events_local']

timepoints_18, timepoints_6 = daylight_times(df['Events_local'])


# In[28]:

### Generate Dashboard

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server

# ------------------------------------------------------------------------------
# App layout


info = dbc.Container(
    [


    dbc.Row(
        dbc.Card(
        [
                dcc.Graph(id='my_topo'),
                dcc.Graph(id='my_route'),
                dcc.Graph(id='my_timeline'),
                dbc.Row(
                [

                dbc.Col(
                dbc.Button('Back'),
                width=1),

                dbc.Col(
                dcc.Slider(
                    id='my-slider',
                    min=0,
 #                   max=(len(list_of_images) - 1),
                    step=1,
   #                 value=0,
                    ),
                    width=10),

                    dbc.Col(
                    dbc.Button('Forward'),
                    width=1),

                    ]
                    ),
                ],
                body=True,
                style={'backgroundColor': '#323130'},
                ),
        ),
#    dbc.Row(
#        dbc.Card(
#            [
#                dcc.Graph(id='my_route'),
#            ]
#            ),
#        ),
#    dbc.Row(
#        dbc.Card(
#            [
#                dcc.Graph(id='my_timeline'),
#            ]
#        ),
#        ),
#    dbc.Row(
#        dbc.Card(
#            [
#                dcc.Slider(
#                    id='my-slider',
#                    min=0,
#                    max=(len(list_of_images) - 1),
#                    step=1,
   #                 value=0,
#                ),
#            ]
#        ),
#        ),
    ],
    style={'backgroundColor': '#323130'}
)

photo_card = dbc.Card(
    [
        dbc.CardImg(id="image"),
    ],
    body=True,
     style={'backgroundColor': '#323130'},
      outline=False,
)


navbar = dbc.Navbar(
    [
        dbc.Col(
            dbc.Nav( dbc.NavItem(dbc.NavLink("Backpacking Dashboard",href='#')), navbar=True),
             lg=3),
        dbc.Col(
            dbc.Nav(dbc.NavItem(dbc.NavLink("Rae Lakes 2020")), navbar=True),
            lg=6),
        dbc.Col(
            dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("More pages", header=True),
                            dbc.DropdownMenuItem("Page 2", href="https://dash.plotly.com/dash-core-components/upload"),
                            dbc.DropdownMenuItem("Page 3", href="https://dash.plotly.com/dash-core-components/upload"),
                        ],
                        #nav=True,
                        in_navbar=True,
                        label="More Trips",
                        color='green',

                    ),

            lg=2),

        dbc.Col(
            dbc.Nav(dbc.NavItem(dbc.NavLink("Github", href='https://github.com/posseward/heroku_backpacking_dashboard')), navbar=True),
            lg=1,
        ),
    ],
    color="green",
    dark=True,
)




app.layout = html.Div(
    [
        dbc.Row(
            dbc.Col(navbar),
                no_gutters=True),

        dbc.Row(
            [
                #dbc.Col( html.Img(id='image' , style={'width' : '100%', 'padding-top' : 20, 'padding-left' : 20}),  width=7),
                dbc.Col(photo_card, md=7),
                dbc.Col(info, md=5),
            ],
        no_gutters=True),

    ],

 style={'backgroundColor':'#323130', 'width' : '100%'},


)


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components


# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
#@app.server.route('{}<image_path>.jpg'.format(static_image_route))
#def serve_image(image_path):
#    image_name = '{}.jpg'.format(image_path)
#    if image_name not in list_of_images:
#        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
#    return flask.send_from_directory(image_directory, image_name)

@app.server.route('{}<image_path>.jpg'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.jpg'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)


@app.callback(
    [dash.dependencies.Output('image', 'src'),
     dash.dependencies.Output('my_topo', 'figure'),
     dash.dependencies.Output('my_timeline', 'figure'),
     dash.dependencies.Output('my_route', 'figure')],
    [dash.dependencies.Input('my_topo', 'clickData'),
     dash.dependencies.Input('my_route', 'clickData'),
     dash.dependencies.Input('my_timeline', 'clickData')])

def update_image_src_map_timeline_route(clickData_map, clickData_route, clickData_timeline ):
#    return static_image_route + list_of_images[value], now just at end



    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 0
    else:

        if ctx.triggered[0]['prop_id'] == 'my-slider.value':
            button_id = button_id = ctx.triggered[0]['value']

        else:
            button_id = ctx.triggered[0]['value']['points'][0]['customdata'][0]



    print(button_id)
    value = button_id


#make map
    dff = df.copy()
    dff['Selected'] = dff['Selected']+0.1
    dff.iloc[value, dff.columns.get_loc('Selected')] = 1


    fig_map = px.scatter_mapbox(df_geo, lat="Latitude", lon="Longitude", opacity=0.5, zoom=zoom, center=center, color="Altitude_feet", hover_data=["Cumulative_distance_miles"])

    fig2_map = px.scatter_mapbox(dff, lat="Latitude", color_discrete_sequence=['white'], lon="Longitude",hover_name="Events", zoom=zoom, hover_data=[], opacity = 0.7, size = "Selected", custom_data=["ID_order"])
    fig_map.add_trace(fig2_map.data[0])





    fig_map.update_layout(mapbox_style= "white-bg",mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ]
            }
        ]
    )
    fig_map.update_layout(height=400, margin={"r":0,"t":0,"l":0,"b":0})
    fig_map.update_layout(coloraxis_showscale=True)

    fig_map.update_layout(coloraxis_colorbar_len=0.5)
    fig_map.update_layout(coloraxis_colorbar_yanchor = 'bottom')
    fig_map.update_layout(coloraxis_colorbar_bgcolor = 'rgba(0, 0, 0, 0)')
    fig_map.update_layout(coloraxis_colorbar_xanchor = 'right')
    fig_map.update_layout(coloraxis_colorbar_xpad = 5)
    fig_map.update_layout(coloraxis_colorbar_ypad = 5)
    fig_map.update_layout(coloraxis_colorbar_x = 1.0)
    fig_map.update_layout(coloraxis_colorbar_y = 0.5)
    fig_map.update_layout(coloraxis_colorbar_tickfont_color = 'white')
    fig_map.update_layout(coloraxis_colorbar_title_font_color = 'white')




#make timeline
    fig_timeline = px.scatter(dff, x='Events_local', y = 'Zeros', color_discrete_sequence=['white'], opacity = 0.7, size = "Selected", height = 200 ,custom_data=["ID_order"])
    fig_timeline.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
    fig_timeline.update_layout(coloraxis_showscale=False)

    fig_timeline.update_yaxes(visible=False, showticklabels=True)
    fig_timeline.update_yaxes(range=[0, 0])
    fig_timeline.update_xaxes(title=None, visible=True, showticklabels=True,ticks="inside")

    for i in range(len(timepoints_6)):
        fig_timeline.add_vrect(x0= timepoints_18[i] ,
                  x1= timepoints_6[i],
                  fillcolor= 'black',
                  opacity=0.4,
                  line_width=0)


    fig_timeline.update_xaxes(range= [ (min(dff['Events_local'])- pd.Timedelta(days=0.25)) , (max(dff['Events_local'])+ pd.Timedelta(days=0.25))   ])

    fig_timeline.update_layout(height=100, margin={"r":0,"t":0,"l":0,"b":0} ,plot_bgcolor='#323930', paper_bgcolor='#323130', font_color="white" )

 #   fig_timeline.update_layout(clickmode='event+select')

#make route

    dff_geo = df_geo.copy()
    #dff_geo['Selected'] = dff_geo['Selected']+0.1
    #dff_geo.iloc[value, dff_geo.columns.get_loc('Selected')] = 1


    fig_route = go.Figure(data=go.Scatter(x=dff_geo.Cumulative_distance_miles, y=dff_geo.Altitude_feet, mode='markers',opacity = 0.9, marker_color= dff_geo.Altitude_feet))

    fig2_route = px.scatter(dff, x="Cumulative_distance_miles", y="Altitude_feet", color_discrete_sequence=['white'], opacity = 0.7, hover_name="Events", hover_data=["Altitude_feet"],
                          size = "Selected", custom_data=["ID_order"])


    fig_route.add_trace(fig2_route.data[0])


    fig_route.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
    fig_route.update_layout(coloraxis_showscale=False)

    fig_route.update_xaxes(title=None, visible=True, showticklabels=True,ticks="inside")



    fig_route.update_yaxes(visible=False, showticklabels=True)
    #fig.update_yaxes(range=[0, 0])
    #fig.update_xaxes(title=None, visible=True, showticklabels=True)

    fig_route.update_layout(height=150, margin={"r":0,"t":0,"l":0,"b":0} ,plot_bgcolor='#323930', paper_bgcolor='#323130', font_color="white")

    fig_route.update_traces(showlegend=False)

    fig_route.update_geos(fitbounds="locations")





#    image_filename = image_directory + '/' +list_of_images[value] # replace with your own image
#    encoded_image = base64.b64encode(open(image_filename, 'rb').read())


#base64 way of serving images
#this goes in the return, isntead of image_filename
#'data:image/png;base64,{}'.format(encoded_image.decode())
#    print(image_filename)

    return static_image_route + list_of_images[value] , fig_map, fig_timeline, fig_route

#mode='jupyterlab'   inline   external

#app.run_server(mode='inline')


# In[29]:



if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)
    
    #for spyder add
    #use_reloader=False


# In[ ]:
