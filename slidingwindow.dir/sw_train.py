#!/anaconda2/bin/python2.7
u"""
sw_train.py
by Yara Mohajerani (09/2018)

Use a sliding window approach to train NN for glacier fronts

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
def load_data(suffix,trn_dir,tst_dir,n_windows,HH,HW):
    #-- make subdirectories for input images
    ddir = {}
    ddir['train'] = trn_dir
    ddir['test'] = tst_dir
    #-- initialize dicttionaries
    images = {} 
    labels = {}
    for d in ['train','test']:
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
        #-- for training data we sample randomly, but for testing data we break down the whole iamge into windows
        if d == 'train':
            labels[d] = np.zeros(n*n_windows*2)
            images[d] = np.zeros((n*n_windows*2, 2*HH + 1 , 2*HW + 1))
            #-- get indices of boundaery for each image and select from them randomly
            for i,f in enumerate(files):
                #-- same file name but different directories for images and labels
                img = np.array(Image.open(os.path.join(subdir,f)).convert('L'))/255.
                lbl = np.array(Image.open(os.path.join(ddir[d],'labels',f.replace('Subset','Front'))).convert('L'))/255.

                ih,iw = np.squeeze(np.nonzero(lbl==0.))
                ih_nb,iw_nb = np.squeeze(np.nonzero(lbl==1.))

                #-- sample twice as many points so we can skip the ones that are close to the edge
                inds_boundary = random.sample(np.arange(0,len(ih)),2*n_windows)
                inds_nb = random.sample(np.arange(0,len(ih_nb)),4*n_windows)
                #-- fill boundary elements
                window_count = 0 #-- window count
                success_count = 0 #-- successful counts
                while success_count < n_windows:
                    j = inds_boundary[window_count]
                    try:
                        images[d][i*n_windows+success_count] = img[ih[j]-HH:ih[j]+HH+1,iw[j]-HW:iw[j]+HW+1]
                        labels[d][i*n_windows+success_count] = 1
                        success_count += 1
                    except:
                        pass
                        #print 'skipped h-index %i and w-index %i'%(ih[j],iw[j])
                    window_count += 1
                #-- fill non-boundary elements
                window_count = 0 #-- window count
                success_count = 0 #-- successful counts
                while success_count < n_windows:
                    j = inds_nb[window_count]
                    #-- if any part of the boundary is in the box, ignore it
                    label_box = lbl[ih_nb[j]-HH:ih_nb[j]+HH+1,iw_nb[j]-HW:iw_nb[j]+HW+1]
                    if np.count_nonzero(label_box==0.) == 0:
                        try:
                            images[d][i*n_windows+success_count] = img[ih_nb[j]-HH:ih_nb[j]+HH+1,iw_nb[j]-HW:iw_nb[j]+HW+1]
                            labels[d][i*n_windows+success_count] = 0
                            success_count += 1
                        except:
                            pass
                            #print 'skipped h-index %i and w-index %i'%(ih_nb[j],iw_nb[j])
                    else:
                        pass
                        #print 'skipped h-index %i and w-index %i for %i boundary points'%(ih_nb[j],iw_nb[j],np.count_nonzero(label_box==0.))
                    window_count += 1

        
        #-- testing data
        else:
            #-- NOTE the following assumes all images have the same dimensions to make things more efficient, but it can
            #-- easily be changed if this is not the case.
            #-- calculate how many windows the whole image breaks down into
            window_h = 2*HH + 1
            window_w = 2*HW + 1
            tot_windows = (img.shape[0] / window_h ) * (img.shape[1] / window_w )
            images[d] = np.zeros((n * tot_windows, window_h , window_w))
            labels[d] = np.zeros(n * tot_windows)
            count = 0
            for i,f in enumerate(files):
                #-- same file name but different directories for images and labels
                img = np.array(Image.open(os.path.join(subdir,f)).convert('L'))/255.
                lbl = np.array(Image.open(os.path.join(ddir[d],'labels',f.replace('Subset','Front'))).convert('L'))/255.

                for ih in range(0,img.shape[0] - window_h, window_h):
                    for iw in range(0,img.shape[1] - window_w, window_w):
                        images[d][count] = img[ih : ih + window_h , iw : iw + window_w]
                        label_box = lbl[ih : ih + window_h , iw : iw + window_w]
                        count_boundary = np.count_nonzero(label_box==0.)
                        #-- if there is any pixel from a boundary, mark as boundary
                        if count_boundary > 0:
                            labels[d][count] = 1
                        else:
                            labels[d][count] = 0
                        count += 1
            if count != n*tot_windows:
                sys.exit('Error in counting. Some test data was not read correctly.')

    print images['train'][0].shape
    print images['test'][0].shape

    return [images,labels]


#-- train model on patches of images
def train_model(parameters):
    glacier = parameters['GLACIER_NAME']
    suffix = parameters['SUFFIX']
    HW = np.int(parameters['HALF_WIDTH']) #-- suggested 10
    HH = np.int(parameters['HALF_HEIGHT']) #-- suggested 10
    n_windows = np.int(parameters['N_WINDOWS'])
    EPOCHS = np.int(parameters['EPOCHS'])
    n_relu = np.int(parameters['N_RELU'])
    n_softmax = np.int(parameters['N_SOFTMAX'])

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ddir = os.path.join(current_dir,'..','%s.dir'%glacier)
    data_dir = os.path.join(ddir, 'data')
    trn_dir = os.path.join(data_dir,'train')
    tst_dir = os.path.join(data_dir,'test')

    #-- load images
    [images,labels] = load_data(suffix,trn_dir,tst_dir,n_windows,HH,HW)


    #-- set up model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(2*HH+1, 2*HW+1)),
        keras.layers.Dense(n_relu, activation=tf.nn.relu),
        keras.layers.Dense(n_softmax, activation=tf.nn.softmax)
    ])

    #-- checkpoint file
    chk_file = os.path.join(ddir,'SW_frontlearn_weights_%iepochs_%iHH_%iHW_%inwindows_%irelu_%isoftmax%s.h5'\
        %(EPOCHS,HH,HW,n_windows,n_relu,n_softmax,suffix))

    #-- if file exists, just read model from file
    if os.path.isfile(chk_file):
        print('Check point exists; loading model from file.')
        # load weights
        model.load_weights(chk_file)

        if parameters['RETRAIN'] in ['y','Y']:
            #-- continue Training
            #-- create checkpoint
            model_checkpoint = keras.callbacks.ModelCheckpoint(chk_file, monitor='loss',\
                verbose=1, save_best_only=True)
            #-- now fit the model
            model.fit(images['train'],labels['train'],epochs=EPOCHS,verbose=1,callbacks=[model_checkpoint])
        else:
            # Compile model
            model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    #-- if not train model
    else:
        print('Did not find check points. Training model...')
        #-- create checkpoint
        model_checkpoint = keras.callbacks.ModelCheckpoint(chk_file, monitor='loss',\
            verbose=1, save_best_only=True)

        # Compile model
        model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        #-- now fit the model
        model.fit(images['train'],labels['train'],epochs=EPOCHS,verbose=1,callbacks=[model_checkpoint])

    print('Model is trained. Running on test data...')

    #-- test accuracy
    test_loss, test_acc = model.evaluate(images['test'], labels['test'])
    print('Test accuracy:', test_acc)


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