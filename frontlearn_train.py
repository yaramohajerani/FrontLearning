#!/usr/bin/env python
u"""
frontlearn_train.py
by Yara Mohajerani (04/2018)

Train U-Net model in frontlearn_unet.py

Update History
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
trn_dir = os.path.join(data_dir,'train')
tst_dir = os.path.join(data_dir,'test')

#-- read in images
def load_train_data():
    #-- get a list of the input files
    trn_list = glob(os.path.join(trn_dir,'images/*.png'))
    tst_list = glob(os.path.join(tst_dir,'images/*.png'))
    #-- get just the file names
    trn_files = [os.path.basename(i) for i in trn_list]
    tst_files = [os.path.basename(i) for i in tst_list]

    #-- read training data
    n = len(trn_files)
    #-- get dimensions, force to 1 b/w channel
    w,h = np.array(Image.open(trn_list[0]).convert('L')).shape

    train_img = np.zeros((n,h,w,1))
    train_lbl = np.zeros((n,w,h,1))
    for i,f in enumerate(trn_files):
        #-- same file name but different directories for images and labels
        train_img[i,:,:,0] = np.array(Image.open(os.path.join(trn_dir,'images',f)).convert('L'))/255.
        train_lbl[i,:,:,0] = np.array(Image.open(os.path.join(trn_dir,'labels',f)).convert('L'))/255.

    #-- also get the test data
    n_test = len(tst_files)
    #-- get dimensions, force to 1 b/w channel
    w_test,h_test = np.array(Image.open(tst_list[0]).convert('L')).shape
    test_img = np.zeros((n_test,h_test,w_test,1))
    for i in range(n_test):
        test_img[i,:,:,0] = np.array(Image.open(tst_list[i]).convert('L'))/255.

    return {'trn_img':train_img,'trn_lbl':train_lbl,'tst_img':test_img,\
        'trn_names':trn_files,'tst_names':tst_files}

def main():
    #-- load images
    data = load_train_data()
    train_img = data['trn_img']
    train_lbl = data['trn_lbl']
    test_img = data['tst_img']

    n,height,width,channels=train_img.shape
    print('width=%i'%width)
    print('height=%i'%height)

    #-- import mod
    unet = imp.load_source('unet_model', os.path.join(ddir,'frontlearn_unet.py'))
    model = unet.unet_model(height,width,channels)

    #-- checkpoint file
    chk_file = os.path.join(ddir,'frontlearn_weights.h5')
    #-- if file exists, just read model from file
    if os.path.isfile(chk_file):
        print('Check point exists; loading model from file.')
        # load weights
        model.load_weights(chk_file)
        # Compile model (required to make predictions)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #-- if not train model
    else:
        print('Did not find check points. Training model...')
        #-- create checkpoint
        model_checkpoint = keras.callbacks.ModelCheckpoint(chk_file, monitor='loss',\
            verbose=1, save_best_only=True)
        #-- now fit the model
        model.fit(train_img, train_lbl, batch_size=10, epochs=50, verbose=1,\
            validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

    print('Model is trained. Running on test data...')

    #-- Now test the model on both the test data and teh train data
    out_imgs = model.predict(train_img, batch_size=1, verbose=1)
    print out_imgs.shape
    #-- save the test image
    for i in range(len(out_imgs)):
        im = image.array_to_img(out_imgs[i])
        im.save(os.path.join(trn_dir,'output/%s'%data['trn_names'][i]))

    out_imgs = model.predict(test_img, batch_size=1, verbose=1)
    print out_imgs.shape
    #-- save the test image
    for i in range(len(out_imgs)):
        im = image.array_to_img(out_imgs[i])
        im.save(os.path.join(tst_dir,'output/%s'%data['tst_names'][i]))

if __name__ == '__main__':
    main()
