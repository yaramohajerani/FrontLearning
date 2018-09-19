#!/usr/bin/env python
u"""
sckikit_canny.py
by Yara Mohajerani (Last Update 09/2018)

Use Canny edge detector from sckikit-image to detect edges

Update History
        09/2018 written
"""
import os
import sys
import numpy as np
from glob import glob
from PIL import Image
from skimage import feature
import matplotlib.pyplot as plt
import scipy.misc
from skimage.morphology import skeletonize
from skimage.future import graph
from skimage import data, segmentation, color, filters, io

#-- read in images
def load_data(suffix,ddir):
    #-- initialize dicttionaries
    images = {} 
    files = {}
    for d in ['train','test']:
        subdir = os.path.join(ddir[d],'images%s'%(suffix))
        #-- get a list of the input files
        file_list = glob(os.path.join(subdir,'*.png'))
        #-- get just the file names
        files[d] = [os.path.basename(i) for i in file_list]
    
        #-- read data
        images[d] = {}
        #-- get indices of boundaery for each image and select from them randomly
        for i,f in enumerate(files[d]):
            #-- same file name but different directories for images and labels
            images[d][i] = np.array(Image.open(os.path.join(subdir,f)).convert('L'))/255.

    return [images,files]

#-- train model and make predictions
def train_model(parameters):
    glacier = parameters['GLACIER_NAME']
    suffix = parameters['SUFFIX']

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.path.join(current_dir,'..','FrontLearning_data')
    glacier_ddir = os.path.join(main_dir,'%s.dir'%glacier)
    data_dir = os.path.join(glacier_ddir, 'data')
    ddir = {}
    ddir['train'] = os.path.join(data_dir,'train')
    ddir['test'] = os.path.join(data_dir,'test')

    #-- load images
    [images,files] = load_data(suffix,ddir)

    sigma = 3
    #-- go through each image and adjust the sigma until a contiuous front is obtained
    for d in ['test']:
        #-- make output directory
        out_subdir = 'output_canny%s'%suffix
        if (not os.path.isdir(os.path.join(ddir[d],out_subdir))):
            os.mkdir(os.path.join(ddir[d],out_subdir))
        #-- make fronts and save to file    
        for i in range(len(images[d])):
            front = feature.canny(images[d][i], sigma=sigma)
            scipy.misc.imsave(os.path.join(ddir[d],out_subdir,'%s'%files[d][i].replace('_Subset',''))\
                , front)



#-- main function to get parameters and pass them along to fitting function
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
            train_model(parameters)

if __name__ == '__main__':
    main()
