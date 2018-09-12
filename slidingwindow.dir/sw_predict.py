#!/anaconda2/bin/python2.7
u"""
sw_predict.py
by Yara Mohajerani (09/2018)

This script uses the trained netwrok from 'sw_train.py' to
draw a contiuous boundary for the glacier front

Update History
    09/2018 Written
"""

import os
import numpy as np
import keras
from keras.preprocessing import image
import imp
import sys
import random
from glob import glob
from PIL import Image
from keras import backend as K
from tensorflow.python.client import device_lib
import tensorflow as tf
#-- Print backend information
print 'device information: ', device_lib.list_local_devices()
print 'available GPUs: ', K.tensorflow_backend._get_available_gpus()

#-- read in images
def load_data(suffix,trn_dir,tst_dir,n_windows_predict,HH,HW):
    #-- make subdirectories for input images
    ddir = {}
    hbnds = {}
    wbnds = {}
    ddir['train'] = trn_dir
    ddir['test'] = tst_dir
    #-- initialize dicttionaries
    images = {} 
    for d in ['test']:#,'train']:
        subdir = os.path.join(ddir[d],'images%s'%(suffix))
        #-- get a list of the input files
        file_list = glob(os.path.join(subdir,'*.png'))
        #-- get just the file names
        files = [os.path.basename(i) for i in file_list]

        #-- read data
        n = len(files)
        #-------------------------------------------------------------------------------------------------------------------------
        #-- now make sliding window data
        #-- note first half is centered on boundaries and the second half is not
        #-------------------------------------------------------------------------------------------------------------------------
        #-- take many samples from the input images to predict the boundary
        #-- keep track of which segments belong to each image
        images[d] = {}
        hbnds[d] = {}
        wbnds[d] = {}
        #-- get indices of boundaery for each image and select from them randomly
        for i,f in enumerate(files):
            images[d][i] = np.zeros((n_windows_predict, 2*HH + 1 , 2*HW + 1))
            hbnds[d][i] = np.zeros((n_windows_predict,2))
            wbnds[d][i] = np.zeros((n_windows_predict,2))
            #-- same file name but different directories for images and labels
            img = np.array(Image.open(os.path.join(subdir,f)).convert('L'))/255.

            #-- take n_windows_predict random samples from the image
            ih = np.random.randint(HH,high=img.shape[0]-HH,size=n_windows_predict)
            iw = np.random.randint(HW,high=img.shape[1]-HW,size=n_windows_predict)
            for j in range(n_windows_predict):
                images[d][i][j] = img[ih[j]-HH:ih[j]+HH+1,iw[j]-HW:iw[j]+HW+1]
                hbnds[d][i][j] = [ih[j]-HH,ih[j]+HH]
                wbnds[d][i][j] = [iw[j]-HW,iw[j]+HW]
                    
    return [images,hbnds,wbnds]


#-- train model on patches of images
def draw_boundary(parameters):
    glacier = parameters['GLACIER_NAME']
    suffix = parameters['SUFFIX']
    HW = np.int(parameters['HALF_WIDTH']) #-- suggested 10
    HH = np.int(parameters['HALF_HEIGHT']) #-- suggested 10
    n_windows = np.int(parameters['N_WINDOWS'])
    n_windows_predict = np.int(parameters['N_WINDOWS_PREDICT'])
    EPOCHS = np.int(parameters['EPOCHS'])
    BATCHES = np.int(parameters['BATCHES'])
    n_relu = np.int(parameters['N_RELU'])
    imb_w = np.int(parameters['IMBALANCE_RATIO'])

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ddir = os.path.join(current_dir,'..','%s.dir'%glacier)
    data_dir = os.path.join(ddir, 'data')
    trn_dir = os.path.join(data_dir,'train')
    tst_dir = os.path.join(data_dir,'test')

    #-- load images
    images,hbnds,wbnds = load_data(suffix,trn_dir,tst_dir,n_windows_predict,HH,HW)


    #-- set up model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(2*HH+1, 2*HW+1)),
        keras.layers.Dense(n_relu, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    #-- checkpoint file
    chk_file = os.path.join(ddir,'SW_frontlearn_weights_%ibtch_%iepochs_%iHH_%iHW_%inwindows_%irelu_%iimbalance%s.h5'\
        %(BATCHES,EPOCHS,HH,HW,n_windows,n_relu,imb_w,suffix))

    #-- if file exists, just read model from file
    if os.path.isfile(chk_file):
        print('Check point exists; loading model from file.')
        # load weights
        model.load_weights(chk_file)
        # Compile model
        model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    #-- if not train model
    else:
        sys.exit('Did not find check points. Need to training model first.')

    #-- make predictions
    for d in ['test']:#,'train']:
        #-- loop through images
        for i in range(len(images[d])):
            pred = model.predict(images[d][i])

            print pred[:,1]
            #-- get the boxes that have boundaries
            ind = np.nonzero(pred[:,1] >= 0.5)

            #-- plot the center of each box that has a boundary
            hcntr = np.mean(hbnds[d][i][ind],axis=1)
            wcntr = np.mean(wbnds[d][i][ind],axis=1)

            print len(hcntr)

            #-- for now just plot centers to see performance
            
            #-- alternate approach: get the average boundary position by doing a weighted average of all the boxes
            #-- since 1 is for boundary and 0 is for no boundary. And this way we take in-between probabilities
            #-- into account.




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
            draw_boundary(parameters)

if __name__ == '__main__':
    main()