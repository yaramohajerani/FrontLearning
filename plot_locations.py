#!/anaconda2/bin/python2.7
u"""
plot_locations.py
by Yara Mohajerani (12/2018)

Make a plot of the locations of the study sites
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Proj,transform
from matplotlib.font_manager import FontProperties
from mpl_toolkits.basemap import Basemap

def main():
    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    headDirectory = os.path.join(current_dir,'..','FrontLearning_data')

    glaciersFolder=os.path.join(headDirectory,'Glaciers')

    glaciers = ['Helheim','Jakobshavn','Kangerlussuaq','Sverdrup']
    h_align = ['left','left','left','left']
    v_space = [2e5,-6e4,2e5,-5e4]

    #-- output projection
    outputCRS = 4326  #(WGS84) #3413
    outProj = Proj(init='epsg:'+str(outputCRS))
    
    #-- initialize figure
    fig, ax1 = plt.subplots(1,1,figsize=(6,6))

    ll = {'lat': [58,82],'lon': [302,20]}
    cent = {'lat': 72.225676431028518,
        'lon': 316.18118891332665}


    #-- ellipsoid parameters for Basemap transformation
    a_axis = 6378137.0#-- [m] semimajor axis of the ellipsoid
    flat = 1.0/298.257223563#-- flattening of the ellipsoid
    rad_e = a_axis*(1.0 -flat)**(1.0/3.0)

    m = Basemap(ax=ax1, projection='stere',lon_0=cent['lon'], lat_0=cent['lat'], \
                lat_ts=70.0, llcrnrlat=ll['lat'][0], urcrnrlat=ll['lat'][1],\
                llcrnrlon=ll['lon'][0], urcrnrlon=ll['lon'][1],\
                rsphere=rad_e, resolution='h', area_thresh=10)


    #-- draw coastlines
    m.drawcoastlines()
   
    #-- set up font properties
    font = FontProperties()
    font.set_weight('bold')
    
    #-- go through each glacier and list the list of months
    for g,a,v in zip(glaciers,h_align,v_space):
        if g == 'Helheim':
            infile = os.path.join(glaciersFolder,g,'%s Image Data_newCorners.csv'%g)
        else:
            infile = os.path.join(glaciersFolder,g,'%s Image Data.csv'%g)
        #-- get corners
        c = pd.read_csv(infile)
        x1,x2,x3,x4 = c['ulX'][0],c['urX'][0],c['lrX'][0],c['llX'][0]
        y1,y2,y3,y4 = c['ulY'][0],c['urY'][0],c['lrY'][0],c['llY'][0]
        #-- get input projection
        inputCRS = c['Projection'][0]
        inProj = Proj(init='epsg:'+str(inputCRS))

        x_mean,y_mean = np.mean([x1,x2,x3,x4]),np.mean([y1,y2,y3,y4])
        x,y = transform(inProj,outProj,x_mean,y_mean)
        x_mean,y_mean = m(x,y)


        ax1.scatter(x_mean,y_mean,s=200,c='dodgerblue',zorder=10)

        #-- set up text location
        if a == 'left':
            x_str = x_mean + 4e4
        else:
            x_str = x_mean - 4e4
        y_str = y_mean - v

        #-- Add glacier name
        ax1.text(x_str, y_str, g, fontproperties=font, verticalalignment='bottom',\
            horizontalalignment=a,color='indigo', fontsize=15)

        
    plt.axis('off')
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='peachpuff',lake_color='aqua')

    m.drawparallels(range(0, 90, 10),labels=[1,0,0,0])
    m.drawmeridians(range(0, 360, 20),labels=[0,0,0,1])

    m.drawmapscale(305, 60, cent['lon'], cent['lat'], 400, barstyle='fancy')
    
    plt.savefig(os.path.join(headDirectory,'Figure_S1-2_map.pdf'),format='pdf')
    plt.close(fig)

if __name__ == '__main__':
    main()