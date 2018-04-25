#!/usr/bin/env python
u"""
frontlearn_postprocess.py
by Yara Mohajerani

Post-Processing of the predictions of the neural network

History
    04/2018 Written
"""
import os
import numpy as np
import keras
from keras.preprocessing import image
import imp
import sys
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

#-- directory setup
#- current directory
ddir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data'))
tst_dir = os.path.join(data_dir,'test')

def post_process(parameters):
    n_batch = np.int(parameters['BATCHES'])
    n_epochs = np.int(parameters['EPOCHS'])
    n_layers = np.int(parameters['LAYERS_DOWN'])
    n_init = np.int(parameters['N_INIT'])
    filter_type = parameters['INPUT_FILTER']
    filter_fraction = np.float(parameters['FILTER_FRACTION'])
    drop = np.float(parameters['DROPOUT'])
    drop_str = ''
    if drop>0:
        drop_str = '_w%.1fdrop'%drop

    #-- total number of layers
    layers_tot = 2*n_layers+1
    #-- make stings for input filters
    if filter_type in ['None','none','NONE','N','n']:
        filter_str = ''
        fraction_str = ''
    else:
        filter_str = '_%s'%filter_type
        fraction_str = '_%.3f'%filter_fraction
    #-- read in output data of the neural network
    ddir = os.path.join(tst_dir,'output_%ibtch_%iepochs_%ilayers_%iinit%s%s%s'\
        %(n_batch,n_epochs,layers_tot,n_init,drop_str,filter_str,fraction_str))

    #-- get a list of the input files
    in_list = glob(os.path.join(ddir,'*.png'))
    n_files = len(in_list)
    w,h = np.array(Image.open(in_list[0]).convert('L')).shape
    #-- read all the files
    for i in range(n_files):
        img = np.array(Image.open(in_list[i]).convert('L'))/255.

        #-- set a threshold for points that are to be identified as the front
        at = 0.9 #-- amplitude threshold
        dt = 2 #-- distance threshold (pixels)
        for ih in range(dt,h-dt):
            for iw in range(dt,w-dt):
                #-- get a sub-image within the distance threshold
                sub_img = img[ih-dt:ih+dt,iw-dt:iw+dt]
                #-- if two points forming a line through the center have values
                #-- less than the treshold, connect the line through the center


]
        #-- write to file
        filename = os.path.basename(in_list[i])
        im = image.array_to_img(img.reshape(h,w,1))
        im.save(os.path.join(ddir,'postprocessed_%s'%filename))

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
