#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:04:07 2020

@author: heather clifford
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression

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


lat = [35.0,38.2]
lon = [-121.7,-117.79]

wf = pd.read_csv('California_MonitoringStations_PM2.5_SimplifiedOutput.csv')
wf['Date'] = pd.to_datetime(wf['Date'])

cmaq = pd.read_csv('California_CMAQ_PM2.5_SimplifiedOutput.csv')
cmaq['Date'] = pd.to_datetime(cmaq['Date'])

date ="2016-01-01"
dates = pd.date_range(start="2016-01-01",end="2016-01-14")
km=12

dataset, yx = gridded_dataset(cmaq,date,km,lat,lon)

lon1=yx[0].T[0]
lat1=yx[1][0]

cmaqpm25=[]
wfpm25=[]

wfd = wf[wf['Date']== pd.Timestamp(date)]

for n,row in wfd.iterrows():
    wfpm25.append(row['PM2.5 Concentration (ug/m3)'])
    
    lonmin = lon1[lon1 >= row['Longitude']].min()
    latmin = lat1[lat1 >= row['Latitude']].min()
    
    lonn = np.where(lon1 == lonmin)[0][0]
    latt = np.where(lat1 == latmin)[0][0]

    cmaqpm25.append(dataset[lonn,latt])


X = np.array(cmaqpm25)
Y = np.array(wfpm25)

Y=np.delete(Y,np.argwhere(np.isnan(X))).reshape(-1, 1)
X=np.delete(X,np.argwhere(np.isnan(X))).reshape(-1, 1)


linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

print(linear_regressor.score(X, Y))
print(linear_regressor.coef_)
print(linear_regressor.intercept_)

fig,ax=plt.subplots()
ax.scatter(X, Y)
ax.set_ylabel('Monitoring Stations')
ax.set_xlabel('CMAQ Data')
ax.set_title('PM2.5 Regression for Jan 1, 2016')
ax.plot(X, Y_pred, color='red')
plt.savefig('LinearReg_PM25_Jan12016.png',dpi=300)
plt.show()









