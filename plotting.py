from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
terrain = sns.color_palette(palette='terrain',n_colors=10)
plasma = sns.color_palette(palette='plasma',n_colors=10)
rainbow = sns.color_palette(palette='rainbow',n_colors=6)

from RentHop import RentHop
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure, show,output_file
from bokeh.sampledata.sample_geojson import geojson
from bokeh.io import output_notebook,output_file
from bokeh.layouts import gridplot,row,column
from bokeh.plotting import figure,show
from bokeh.io import output_file, show
from bokeh.models import (
    GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)



mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']



feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]



train_df = pd.read_json("input/train.json")
test_df = pd.read_json("input/test.json")

def violinBedroomsBathrooms(train_df):
    train_df['bathrooms'].ix[train_df['bathrooms'] <= 0] = 1
    train_df['bedrooms'].ix[train_df['bedrooms'] <= 0] = 1

    train_df['bathrooms'].ix[train_df['bathrooms'] > 3] = 3
    train_df['bedrooms'].ix[train_df['bedrooms'] > 3] = 3

    train_df['bedrooms_bathrooms'] = train_df['bedrooms'] * 10.0 + train_df['bathrooms']
    print(train_df['bedrooms_bathrooms'])
    # plt.figure(figsize=(8,4))

    sns.violinplot(x='interest_level', y='bedrooms_bathrooms', data=train_df)
    plt.show()
    
def barBedroomsBathrooms(df):
    df['bathrooms'].ix[df['bathrooms'] <= 0] = 1
    df['bedrooms'].ix[df['bedrooms'] <= 0] = 1

    df['bathrooms'].ix[df['bathrooms'] > 3] = 3
    df['bedrooms'].ix[df['bedrooms'] > 3] = 3

    df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']

    sns.countplot(x='bedrooms_bathrooms', hue='interest_level',data=df)
    plt.show()

def barBedroomsBathroomsPrice(df):
    df.loc[df['bathrooms'] <= 0,'bathrooms'] = 1
    df.loc[df['bedrooms'] <= 0,'bedrooms'] = 1

    df.loc[df['bathrooms'] > 3,'bathrooms'] = 3
    df.loc[df['bedrooms'] > 3,'bedrooms'] = 3

    df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']

    df = df[df['price'] < 10000]

    trans = {
        'low':'red',
        'medium': 'green',
        'high':'blue'
    }
    colors = [trans[x.interest_level] for i,x in df.iterrows()]


    plt.scatter(df.price, df.bedrooms_bathrooms, c=colors)

    #plt.yticks(np.arange(0,40,1))
    plt.yticks([11,12,13,14,15,21,22,23,24,31,32,33,34])
    plt.show()

def location(df, distance = 3):
    def pointTopoint(a,b):
        a = (np.radians(a[0]),np.radians(a[1]))
        b = (np.radians(b[0]), np.radians(b[1]))

        dlon = b[1] - a[1]
        dlat = b[0] - a[0]
        a = np.power(np.sin(dlat / 2), 2) + np.cos(a[0]) * np.cos(b[0]) * np.power((np.sin(dlon / 2)), 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = 3961 * c
        return d

    bronx_center = (40.834600, -73.879082)
    extreme_point = (40.859532, -73.913929)

    #distance_from_center = pointTopoint(bronx_center,extreme_point)

    df['distance'] = pointTopoint((df.latitude,df.longitude),bronx_center)
    df = df[pointTopoint(bronx_center,(df.latitude,df.longitude)) <= distance]

    return df

def mapa(df):
    p = figure(title="interest level based on geography", y_range=(40.65, 40.85), x_range=(-74.05, -73.85))
    p.xaxis.axis_label = 'longitude'
    p.yaxis.axis_label = 'latitude'
    lowLat = df['latitude'][df['interest_level'] == 'low']
    lowLong = df['longitude'][df['interest_level'] == 'low']
    medLat = df['latitude'][df['interest_level'] == 'medium']
    medLong = df['longitude'][df['interest_level'] == 'medium']
    highLat = df['latitude'][df['interest_level'] == 'high']
    highLong = df['longitude'][df['interest_level'] == 'high']
    p.circle(lowLong, lowLat, size=3, color=terrain.as_hex()[1], fill_alpha=0.1, line_alpha=0.1, legend='low')
    p.circle(medLong, medLat, size=3, color=plasma.as_hex()[9], fill_alpha=0.1, line_alpha=0.1, legend='med')
    p.circle(highLong, highLat, size=3, color=plasma.as_hex()[5], fill_alpha=0.1, line_alpha=0.1, legend='high')
    show(p, notebook_handle=False)

def mapaTest(df):

    #df = df.head(100)
    map_options = GMapOptions(lat=40.75, lng=-74.00, map_type="roadmap", zoom=11)

    plot = GMapPlot(
        x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options
    )
    plot.title.text = "New York"

    plot.api_key = "AIzaSyAJRzHu-dh6vUh7lmoUdCZtMr1xFqo9FVU"

    #high
    sourceHigh = ColumnDataSource(
        data=dict(
            lat=df[df['interest_level'] == 'high'].latitude,
            lon=df[df['interest_level'] == 'high'].longitude,
        )
    )
    circleHigh = Circle(x="lon", y="lat", size=2, fill_color="#000099", fill_alpha=0.8, line_color=None)
    # medium
    sourceMedium = ColumnDataSource(
        data=dict(
            lat=df[df['interest_level'] == 'medium'].latitude,
            lon=df[df['interest_level'] == 'medium'].longitude,
        )
    )
    circleMedium = Circle(x="lon", y="lat", size=2, fill_color="yellow", fill_alpha=0.8, line_color=None)
    # low
    sourceLow = ColumnDataSource(
        data=dict(
            lat=df[df['interest_level'] == 'low'].latitude,
            lon=df[df['interest_level'] == 'low'].longitude,
        )
    )
    circleLow = Circle(x="lon", y="lat", size=2, fill_color="red", fill_alpha=0.8, line_color=None)

    #plot.add_glyph(sourceHigh, circleHigh)
    #plot.add_glyph(sourceMedium, circleMedium)
    plot.add_glyph(sourceLow, circleLow)

    plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
    output_file("gmap_plot.html")
    show(plot)

def correlation(df):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True, n=100)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,
                square=True, xticklabels=True, yticklabels=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.show()

def featureImportance(df):
    from sklearn.ensemble import RandomForestClassifier
    rhop = RentHop()
    train_X, train_Y, test_X = rhop.getData(train_df,test_df)
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train_X,train_Y)
    pd.Series(index = rhop.features_to_use,data = rf.feature_importances_).sort_values().plot(kind='bar')
    plt.show()



featureImportance(train_df)
#mapaTest(train_df)
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
# print(train_df[train_df['interest_level']=='high'].price.describe())
# print(train_df[train_df['interest_level']=='medium'].price.describe())
# print(train_df[train_df['interest_level']=='low'].price.describe())

#print(train_df.groupby('manager_id').count())