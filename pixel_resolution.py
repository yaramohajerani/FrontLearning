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
            print('\n\nNN x error %f pixels'%(96.31/(dx/200.)))
            print('NN y error %f pixels'%(96.31/(dy/300.)))

            print('\nSobel x error %f pixels'%(836.35/(dx/200.)))
            print('Sobel y error %f pixels'%(836.35/(dy/300.)))

            print('\nManual x error %f pixels'%(92.49/(dx/200.)))
            print('Manual y error %f pixels'%(92.49/(dy/300.)))

            print('\nsuccess NN x error %f pixels'%(85.26125/(dx/200.)))
            print('success NN y error %f pixels'%(85.26125/(dy/300.)))

            print('\nsuccess Sobel x error %f pixels'%(193.01875/(dx/200.)))
            print('success Sobel y error %f pixels'%(193.01875/(dy/300.)))

            print('\nsuccess Manual x error %f pixels'%(89.095/(dx/200.)))
            print('success Manual y error %f pixels'%(89.095/(dy/300.)))

        elif glacier == 'Sverdrup':
            print('\n\nx error %f pixels'%(137.61/(dx/200.)))
            print('y error %f pixels'%(137.61/(dy/300.)))
        elif glacier == 'Kangerlussuaq':
            # 175.90 with 500m buffer but edges are mismatched because of weird boundary
            # so 700 buffer might be a fairer comparison
            print('\n\nx error %f pixels'%(124.40/(dx/200.))) 
            print('y error %f pixels'%(124.40/(dy/300.)))
        print('\n')        
