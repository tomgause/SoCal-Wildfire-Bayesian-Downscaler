#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:04:07 2020

@author: heather clifford
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import math
import numpy as np
from scipy.interpolate import griddata

def plot_rectangle(bmap, lonmin,lonmax,latmin,latmax):
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    bmap.plot(xs, ys,latlon = True)
    
def distance_between_lat_lon(lat,lon):
    #calculate distance between latitude and longitude in km
    r = 6373
    dlon = math.radians(lon[1]) - math.radians(lon[0])
    dlat = math.radians(lat[1]) - math.radians(lat[0])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat[0])) * math.cos(math.radians(lat[1])) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = r * c 
    return d

def gridded_dataset(data,date,km,lat,lon):
    
    dd = data[data['Date']== pd.Timestamp(date)]
     
    latdis = distance_between_lat_lon(lat,[lon[0],lon[0]])
    londis = distance_between_lat_lon([lat[0],lat[0]],lon)
    
    latm = np.linspace(lat[0],lat[1],int(latdis/km))
    lonm = np.linspace(lon[0],lon[1],int(londis/km))
    
    xi, yi = np.meshgrid(latm, lonm)

    x=dd['Latitude'].values.tolist()
    y=dd['Longitude'].values.tolist()
    z=dd['PM2.5 Concentration (ug/m3)'].values.tolist()
    
    zi = griddata((x,y), z, (xi, yi))
    
    return zi,(yi,xi)

def plot_griddeddata(data,date,km,lat,lon):
    dataset, yx = gridded_dataset(cmaq,date,km,lat,lon)
    
    m = Basemap(projection='lcc', resolution='h',lat_0=37.5, lon_0=-119, width=1E6, height=1.2E6)
    m_lon, m_lat = m(*yx)
    
    fig, ax = plt.subplots(figsize=(6, 12))
    m.drawcountries(color='gray')
    m.drawcoastlines(linewidth=0.25)
    m.drawstates(color='gray')
    m.contourf(m_lon, m_lat, dataset, 500, cmap='coolwarm', zorder = 2)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('PM2.5 Concentration (ug/m3)')
    
    plot_rectangle(m,lon[0],lon[1], lat[0],lat[1])
    
    plt.title('{} PM2.5 at {}km resolution on {}'.format(data.loc[0,'Dataset'],km,date))
    plt.tight_layout()
    plt.savefig('{}_PM2.5_{}km_{}.png'.format(data.loc[0,'Dataset'],km,date),dpi=300)
    plt.show()


lat = [35.0,38.2]
lon = [-121.7,-117.79]

wf = pd.read_csv('California_MonitoringStations_PM2.5_SimplifiedOutput.csv')
wf['Date'] = pd.to_datetime(wf['Date'])

cmaq = pd.read_csv('California_CMAQ_PM2.5_SimplifiedOutput.csv')
cmaq['Date'] = pd.to_datetime(cmaq['Date'])

date ="2016-01-01"
km=12

dataset, yx = gridded_dataset(cmaq,date,km,lat,lon)

plot_griddeddata(cmaq,date,km,lat,lon)

