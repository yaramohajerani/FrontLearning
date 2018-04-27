#!/usr/bin/env python
u"""
frontlearn_postprocess.py
by Yara Mohajerani

Post-Processing of the predictions of the neural network

TO BE COMPLETED

History
    04/2018 Written
"""
import os
import numpy as np
import imp
import sys
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

#-- directory setup
#- current directory
ddir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data'))
tst_dir = os.path.join(data_dir,'test')

def post_process(parameters):
    n_batch = int(parameters['BATCHES'])
    n_epochs = int(parameters['EPOCHS'])
    n_layers = int(parameters['LAYERS_DOWN'])
    n_init = int(parameters['N_INIT'])
    sharpness = float(parameters['SHARPNESS'])
    contrast = float(parameters['CONTRAST'])
    drop = float(parameters['DROPOUT'])
    drop_str = ''
    if drop>0:
        drop_str = '_w%.1fdrop'%drop

    #-- total number of layers
    layers_tot = 2*n_layers+1

    if sharpness in ['None','none','NONE','N','n']:
        sharpness_str = ''
    else:
        sharpness_str = '_sharpness%.1f'%sharpness
    if contrast in ['None','none','NONE','N','n']:
        contrast_str = ''
    else:
        contrast_str = '_contrast%.1f'%contrast

    #-- read in output data of the neural network
    ddir = os.path.join(tst_dir,'output_%ibtch_%iepochs_%ilayers_%iinit%s%s%s'\
        %(n_batch,n_epochs,layers_tot,n_init,drop_str,sharpness_str,contrast_str))

    #-- get a list of the input files
    in_list = glob(os.path.join(ddir,'*.png'))
    n_files = len(in_list)
    w,h = np.array(Image.open(in_list[0]).convert('L')).shape
    mask = None
    #-- vectorize files
    for i in range(n_files):
        with rasterio.drivers():
            with rasterio.open(in_list[i]) as src:
                image = src.read(1) # first band
                results = ({'properties': {'raster_val': v}, 'geometry': s}\
                    for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.affine)))

                geoms = list(results)
                print(len(geoms))
                f = plt.figure(i)
                for j in range(len(geoms)):
                    s = shape(geoms[j]['geometry'])
                    x,y = s.exterior.xy
                    plt.plot(x,y)
                plt.savefig(os.path.join(ddir,in_list[i][:-4]+'.pdf'),format='pdf')

#-- main function to get parameters and pass them along to the postprocessing function
def main():
    if (len(sys.argv) == 1):
        sys.exit('You need to input at least one parameter file to set run configurations.')
    else:
        #-- Input Parameter Files (sys.argv[0] is the python code)
        input_files = sys.argv[1:]
        #-- for each input parameter file
        for file in input_files:
            #-- keep track of progress
            print(os.path.basename(file))
            #-- variable with parameter definitions
            parameters = {}
            #-- Opening parameter file and assigning file ID number (fid)
            fid = open(file, 'r')
            #-- for each line in the file will extract the parameter (name and value)
            for fileline in fid:
                #-- Splitting the input line between parameter name and value
                part = fileline.split()
                #-- filling the parameter definition variable
                parameters[part[0]] = part[1]
            #-- close the parameter file
            fid.close()

            #-- pass parameters to training function
            post_process(parameters)

if __name__ == '__main__':
    main()
