u"""
pixel_resolution

Get pixel resolution for each fjord
"""
import os
import numpy as np
import pandas as pd

#-- directory setup
#- current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
headDirectory = os.path.join(current_dir,'..','FrontLearning_data')

glaciersFolder=os.path.join(headDirectory,'Glaciers')

glacier_list = ['Helheim','Jakobshavn','Kangerlussuaq','Sverdrup']

for glacier in glacier_list:
    print('Glacier: %s'%glacier)
    infile = os.path.join(glaciersFolder,glacier,'%s Image Data.csv'%glacier)
    d = pd.read_csv(infile)

    ind_list = [0]#,8] #np.arange(len(d['urX']))

    for ind in ind_list:

        dx = np.sqrt((d['ulX'][ind] - d['urX'][ind])**2 + (d['ulY'][ind] - d['urY'][ind])**2)
        dy = np.sqrt((d['urX'][ind] - d['lrX'][ind])**2 + (d['urY'][ind] - d['lrY'][ind])**2)

        print('x pixel size %f'%(dx/200.))
        print('y pixel size %f'%(dy/300.))

        if glacier == 'Helheim':
            print('x error %f pixels'%(96.31/(dx/200.)))
            print('y error %f pixels'%(96.31/(dy/300.)))
        elif glacier == 'Sverdrup':
            print('x error %f pixels'%(143.24/(dx/200.)))
            print('y error %f pixels'%(143.24/(dy/300.)))
        elif glacier == 'Kangerlussuaq':
            # 175.90 with 500m buffer but edges are mismatched because of weird boundary
            # so 700 buffer might be a fairer comparison
            print('x error %f pixels'%(124.40/(dx/200.))) 
            print('y error %f pixels'%(124.40/(dy/300.)))
        print('\n')        
