#!/anaconda2/bin/python2.7
u"""
plot_seasonality.py
by Yara Mohajerani (12/2018)

Make a plot of the timestamps of the training and test data
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


#-- directory setup
#- current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
headDirectory = os.path.join(current_dir,'..','FrontLearning_data')

glaciersFolder=os.path.join(headDirectory,'Glaciers')

glaciers = ['Helheim','Jakobshavn','Kangerlussuaq','Sverdrup']

months = {}

#-- go through each glacier and list the list of months
for g in glaciers:
    infile = os.path.join(glaciersFolder,g,'%s Image Data.csv'%g)
    #-- get file names
    filenames = pd.read_csv(infile)['Image File']
    #-- initialize list of months for each glacier
    months[g] = np.zeros(len(filenames),dtype=int)
    #-- go through files and extract months
    for i,f in enumerate(filenames):
        date_str = f.split('_')[3]
        
        yr = int(date_str[:4])
        mn = int(date_str[4:6])
        dy = int(date_str[6:8])

        months[g][i] = mn

#-- now make plot of months
fig, ax = plt.subplots(1,1,figsize=(6,3))

for l,g in enumerate(glaciers):
    #-- go through each month and make the size of the scatter point propotional
    #-- to the number of images from that month
    x = np.arange(1,13) #x-axis
    count = np.zeros(len(x))
    for i in x:
        count[i-1] = np.count_nonzero(months[g]==i)
    
    #-- normalize (out of 200)
    count = count/np.max(count) * 200

    #-- plot months
    ax.scatter(x,np.ones(len(x))*l, s = count) #s = np.sqrt(count)*100)

mon_lbls = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.xticks(x, mon_lbls, rotation=45)
plt.yticks(np.arange(len(glaciers)), glaciers)
plt.grid(True)
plt.subplots_adjust(bottom=0.15,left=0.2)
plt.savefig(os.path.join(headDirectory,'Figure_S1-1.pdf'),format='pdf')
plt.close(fig)