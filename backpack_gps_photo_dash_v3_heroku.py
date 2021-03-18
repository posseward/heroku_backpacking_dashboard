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


# ### test

# In[2]:


#https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows
#vectorized haversine function, for determining the distance between two locations on earth


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 +         np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


# In[3]:


#https://stackoverflow.com/questions/63787612/plotly-automatic-zooming-for-mapbox-maps
#Based off a collection of coordinates, finds optimal zoom and centering for a plotly mapbox


def zoom_center(lons: tuple=None, lats: tuple=None, lonlats: tuple=None,
                format: str='lonlat', projection: str='mercator',
                width_to_height: float=2.0) -> (float, dict):

    """Finds optimal zoom and centering for a plotly mapbox.
    Must be passed (lons & lats) or lonlats.
    Temporary solution awaiting official implementation, see:
    https://github.com/plotly/plotly.js/issues/3434

    Parameters
    --------
    lons: tuple, optional, longitude component of each location
    lats: tuple, optional, latitude component of each location
    lonlats: tuple, optional, gps locations
    format: str, specifying the order of longitud and latitude dimensions,
        expected values: 'lonlat' or 'latlon', only used if passed lonlats
    projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.

    Returns
    --------
    zoom: float, from 1 to 20
    center: dict, gps position with 'lon' and 'lat' keys

    >>> print(zoom_center((-109.031387, -103.385460),
    ...     (25.587101, 31.784620)))
    (5.75, {'lon': -106.208423, 'lat': 28.685861})
    """
    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError(
                'Must pass lons & lats or lonlats'
            )

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        'lon': round((maxlon + minlon) / 2, 6),
        'lat': round((maxlat + minlat) / 2, 6)
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])

    if projection == 'mercator':
        margin = 1.2
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(
            f'{projection} projection is not implemented'
        )

    return zoom, center


# In[4]:

#enter path to your folder of images
path_imagefolder = 'rae_lakes_github'

# In[7]:

#get data from images (coordinates and timepoints) and image file names
def create_image_df(path_imagefolder):
    
        
    image_list = os.listdir(path_imagefolder)
    image_list = [a for a in image_list if a.endswith('jpg')]
    data_list = []
    ID_list = []
    for i, a in enumerate(image_list):
        data = gpsphoto.getGPSData(path_imagefolder + f'/{a}')
        data_list.append(data)
        ID_list.append(a)
    
    #create dataframe
    df = pd.DataFrame(data_list)
    df['ID_list'] = ID_list
    
    #convert altitude meters of feet
    df['Altitude_feet'] = df['Altitude'] * 3.28084

    
    #drop NA columns
    df.dropna(inplace=True)
    
    #reset index for NA
    df.reset_index(drop=True, inplace=True)
    
    df_events = []
    
    for event in range(len(df)):
        date_event = datetime.strptime(df['Date'][event], "%m/%d/%Y")
        time_event = datetime.strptime(df['UTC-Time'][event], "%H:%M:%S")
        combine_event = datetime.combine(datetime.date(date_event), datetime.time(time_event))
        df_events.append(combine_event)
    
    #create data column of timepoints
    df['Events'] = df_events
    
    #sort dataframe by timepoints
    df = df.sort_values(by='Events')
    df.reset_index(inplace=True)
    
    #create data columns of selected and not selected for dashboard
    df['Zeros'] = 0
    df['Selected'] = 0
    
    #normalize timestamps
    scaler = MinMaxScaler()
    df['Events_normalized'] = scaler.fit_transform(df['Events'].values.reshape(-1,1))
    
    #difference between each timestamp
    df['Events_difference'] = df.Events_normalized - df.Events_normalized.shift()
    df['Events_difference'][0] = 0
    df['ID_order'] = np.arange(len(df))
    
    #convert events to universial time??
    df['Events'] = df['Events'].dt.tz_localize('UTC')
    
    df['Events_local'] = df['Events'].dt.tz_convert('US/Pacific')
        
    return df
    
# In[]
df = create_image_df(path_imagefolder)

# In[11]:

#generate static image for display on website
image_directory = path_imagefolder
list_of_images = df['ID_list'].tolist()

static_image_route = '/static/'

# In[12]:

#get geojson file
geo_filename = 'rae_lakes_2020.geojson'


# In[13]:

#go get the GPS track, probably move this up to the beginning with getting the other data

#now you go get file in the filename GUI
#df_gps = geopandas.read_file(r"C:\Users\peter\Desktop\datascience career\trinity_alps_2020.geojson")

df_gps = geopandas.read_file(geo_filename)


combined = []

for a in range(len(df_gps.geometry[0])):
    coords=list(df_gps.geometry[0][a].coords)
    combined += coords

df_geo = pd.DataFrame(combined, columns =['Longitude', 'Latitude', 'Altitude'])
df_geo['Altitude_feet'] = df_geo['Altitude'] * 3.28084


# In[14]:

# find distance between each coordinate on route
df_geo['Distance'] = haversine(df_geo.Latitude.shift(), df_geo.Longitude.shift(),
                 df_geo.Latitude, df_geo.Longitude)

#convert km to miles and its good
#add cumulative distance

#convert to miles
df_geo['Distance_miles'] = df_geo['Distance'] * 0.621371

#calculate cumulative distance
df_geo['Cumulative_distance_miles']= df_geo['Distance_miles'].cumsum()

#create data column of the index of data points
df_geo['Route_order'] = np.arange(len(df_geo))

#enter in zero for initial coordinate, I assume because it can't subtract from anything
df_geo.Distance[0] = 0
df_geo.Distance_miles[0] = 0
df_geo.Cumulative_distance_miles[0] = 0

#create a normalized distance
df_geo['Cumulative_distance_normalized'] = df_geo.Cumulative_distance_miles / df_geo.Distance_miles.sum()


# In[15]:

#find zoom and center for map
zoom, center = zoom_center(
    lons= list(df_geo.Longitude),
    lats= list(df_geo.Latitude),
    width_to_height = 4.0
)

# In[16]:

#https://stackoverflow.com/questions/47534715/get-nearest-point-from-each-other-in-pandas-dataframe

#find the closest point to another in a dataframe



mat = scipy.spatial.distance.cdist(df_geo[['Latitude','Longitude']],
                              df[['Latitude','Longitude']], metric='euclidean')


# In[17]:


#create data frame with index as each route coordinate and columns as picture coordinates and each value being the distance
#between the the two
new_df = pd.DataFrame(mat, index=df_geo['Route_order'], columns=df['ID_order'])
#new_df


# In[18]:


#create a dictionary list of the dataframe columns, find ten cloest
dict = {}


# creating a list of dataframe columns
columns = list(new_df)

for i in columns:
    test = new_df.sort_values(by=[i])
    dict[i] =test[i].index[0:10]

    #find 10 closest

# In[19]:
#transpose it
df_close = pd.DataFrame(dict)
df_close = df_close.T

# In[22]:

previous_location = df_geo.iloc[df_close.iloc[0]].min().Cumulative_distance_normalized
#previous_location


# In[23]:

# creating a list of dataframe columns
distance_difference = []
best_index = []
best_spacetime = []
best_cumulative_time = []

previous_location = df_geo.iloc[df_close.iloc[0].min()].Cumulative_distance_normalized

# creating a list of dataframe columns
i = range(len(df_close))

for i in columns:

    distance_difference.append( (df_geo.iloc[df_close.iloc[i]].Cumulative_distance_normalized - previous_location ))
    #print(distance_difference[i])
    temporal_difference = df['Events_difference'][i]
    #print(temporal_difference)
    spacetime = (temporal_difference - (distance_difference[i]))
    #print(spacetime)
    df_spacetime = pd.DataFrame(spacetime)
    #print(df_spacetime)

    best_index.append (   int(df_spacetime.abs().idxmin()) )
    #print(best_index)
    best_spacetime.append ( float(df_spacetime.abs().min() ))
    #print(best_spacetime)

    best_cumulative_time.append(df_geo['Cumulative_distance_miles'][best_index[i]])

    #best_index.append(spacetime.index[min(spacetime, key=abs)])
    #best_spacetime.append( min(spacetime, key=abs) )

    previous_location = df_geo.iloc[best_index[i]].Cumulative_distance_normalized
    #print(previous_location)


# In[24]:


df['Cumulative_distance_miles'] = best_cumulative_time


# In[25]:

#for each column, find 5 lowest distance values. Then find the 5 corresponding cumulative distances
#for each column, find the its increamental increase in normalized time
#for each column subtract the current cumulative distance from the previous one. To get distance increase.
#I guess for the first timepoint the distance increase is just from 0, as in the time increase.
#Compare distance increase to time increase, then pick accordingly


# In[26]:


test = df['Events_local']

def daylight_times(test):
    start = (min(test)).round(freq = 'D') - pd.Timedelta(days=2.25)
    end = (max(test)).round(freq = 'D') + pd.Timedelta(days=2.25)

    timepoints_18 = pd.date_range(start=start, end= (end + pd.Timedelta(days=0.5)), freq= '24H')
    timepoints_6 = pd.date_range(start= (start + pd.Timedelta(days=0.5)) , end=end, freq= '24H')

    return timepoints_18,timepoints_6

timepoints_18, timepoints_6 = daylight_times(test)


# In[28]:

### Generate Dashboar

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
    app.run_server(debug=True)
    
    #for spyder add
    #use_reloader=False


# In[ ]:
