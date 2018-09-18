#!/usr/bin/env python
u"""
crop_input.py
by Yara Mohajerani (09/2018)

Crop input data.

Update History
        09/2018 Written
"""
import os
import numpy as np
import imp
import sys
from glob import glob
from PIL import Image,ImageOps
from keras.preprocessing import image

#-- read in images
def load_data(suffix,ddir,hcrop,wcrop):
    files = {}
    images = {}
    labels = {}
    for d in ['train','test']:
        #-- make subdirectories for input images
        subdir = os.path.join(ddir[d],'images%s'%(suffix))
        #-- get a list of the input files
        file_list = glob(os.path.join(subdir,'*.png'))
        #-- get just the file names
        files[d] = [os.path.basename(i) for i in file_list]

        #-- read training data
        n = len(files[d])
        #-- get dimensions, force to 1 b/w channel
        im_shape = np.array(Image.open(file_list[0]).convert('L')).shape
        h,w = im_shape
        
        images[d] = np.ones((n,h-2*hcrop,w-2*wcrop))
        labels[d] = np.ones((n,h-2*hcrop,w-2*wcrop))
        for i,f in enumerate(files[d]):
            #-- same file name but different directories for images and labels
            img = np.array(Image.open(os.path.join(subdir,f)).convert('L'))/255.
            lbl = np.array(Image.open(os.path.join(ddir[d],'labels',f.replace('Subset','Front'))).convert('L'))/255.

            images[d][i][:,:] = img[hcrop:h-hcrop,wcrop:w-wcrop]
            labels[d][i][:,:] = lbl[hcrop:h-hcrop,wcrop:w-wcrop]

        images[d] = images[d].reshape(n,h-2*hcrop,w-2*wcrop,1)
        labels[d] = labels[d].reshape(n,h-2*hcrop,w-2*wcrop,1)

    return [images,labels,files]

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
    images,labels,files = load_data(suffix,ddir,30,25)

    #-- make output directory
    for d in ['train','test']:
        out_subdir_img = os.path.join(ddir[d],'images%s_cropped'%suffix)
        out_subdir_lbl = os.path.join(ddir[d],'labels_cropped')
        if (not os.path.isdir(out_subdir_img)):
            os.mkdir(out_subdir_img)
        if (not os.path.isdir(out_subdir_lbl)):
            os.mkdir(out_subdir_lbl)
        #-- save the cropped image
        for i in range(len(files[d])):
            im = image.array_to_img(images[d][i])
            lb = image.array_to_img(labels[d][i])
            im.save(os.path.join(out_subdir_img,'%s'%files[d][i]))
            lb.save(os.path.join(out_subdir_lbl,'%s'%files[d][i].replace('Subset','Front')))

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
