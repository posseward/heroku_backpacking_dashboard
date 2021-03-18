# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:39:44 2021

@author: peter
"""
import os
from GPSPhoto import gpsphoto
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import geopandas
import scipy

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
    
# In[13]:

def create_geo_df(geo_filename):
    
    df_gps = geopandas.read_file(geo_filename)
    
    combined = []
    
    for a in range(len(df_gps.geometry[0])):
        coords=list(df_gps.geometry[0][a].coords)
        combined += coords
    
    df_geo = pd.DataFrame(combined, columns =['Longitude', 'Latitude', 'Altitude'])
    df_geo['Altitude_feet'] = df_geo['Altitude'] * 3.28084
    
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

    return df_geo

# In[16]:

#https://stackoverflow.com/questions/47534715/get-nearest-point-from-each-other-in-pandas-dataframe

#find the closest point to another in a dataframe


def predict_image_route_distance(df,df_geo):
    
#for each column, find 5 lowest distance values. Then find the 5 corresponding cumulative distances
#for each column, find the its increamental increase in normalized time
#for each column subtract the current cumulative distance from the previous one. To get distance increase.
#I guess for the first timepoint the distance increase is just from 0, as in the time increase.
#Compare distance increase to time increase, then pick accordingly

    
    mat = scipy.spatial.distance.cdist(df_geo[['Latitude','Longitude']],
                                  df[['Latitude','Longitude']], metric='euclidean')
    
    #create data frame with index as each route coordinate and columns as picture coordinates and each value being the distance
    #between the the two
    new_df = pd.DataFrame(mat, index=df_geo['Route_order'], columns=df['ID_order'])
    #new_df
    
    #create a dictionary list of the dataframe columns, find ten cloest
    dict = {}
    
    
    # creating a list of dataframe columns
    columns = list(new_df)
    
    for i in columns:
        test = new_df.sort_values(by=[i])
        dict[i] =test[i].index[0:10]
    
        #find 10 closest
    
    #transpose it
    df_close = pd.DataFrame(dict)
    df_close = df_close.T
    
    previous_location = df_geo.iloc[df_close.iloc[0]].min().Cumulative_distance_normalized
    #previous_location
    
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
    
    df['Cumulative_distance_miles'] = best_cumulative_time
    
    return df

# In[26]:
def daylight_times(test):
    start = (min(test)).round(freq = 'D') - pd.Timedelta(days=2.25)
    end = (max(test)).round(freq = 'D') + pd.Timedelta(days=2.25)

    timepoints_18 = pd.date_range(start=start, end= (end + pd.Timedelta(days=0.5)), freq= '24H')
    timepoints_6 = pd.date_range(start= (start + pd.Timedelta(days=0.5)) , end=end, freq= '24H')

    return timepoints_18,timepoints_6
