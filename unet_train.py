#!/usr/bin/env python
u"""
frontlearn_train_AUGMENT.py
by Yara Mohajerani (Last Update 09/2018)

Train U-Net model in frontlearn_unet.py

Update History
        09/2018 Combine with original script and clean up
        05/2018 Forked from frontlearn_train.py
"""
import os
import numpy as np
import keras
from keras.preprocessing import image
import imp
import sys
from glob import glob
from PIL import Image,ImageOps
from keras import backend as K
from tensorflow.python.client import device_lib
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#-- Print backend information
print(device_lib.list_local_devices())
print(K.tensorflow_backend._get_available_gpus())

#-- read in images
def load_data(suffix,trn_dir,tst_dir,n_layers,augment,crop_str):
    #-- make subdirectories for input images
    trn_subdir = os.path.join(trn_dir,'images%s%s'%(suffix,crop_str))
    tst_subdir = os.path.join(tst_dir,'images%s%a'%(suffix,crop_str))
    #-- get a list of the input files
    trn_list = glob(os.path.join(trn_subdir,'*.png'))
    tst_list = glob(os.path.join(tst_subdir,'*.png'))
    #-- get just the file names
    trn_files = [os.path.basename(i) for i in trn_list]
    tst_files = [os.path.basename(i) for i in tst_list]

    #-- read training data
    n = len(trn_files)
    if augment:
        #-- need to triple for the extra two augmentations
        n *= 3
    #-- get dimensions, force to 1 b/w channel
    im_shape = np.array(Image.open(trn_list[0]).convert('L')).shape
    h,w = im_shape
    # pad height and width until it's at least divisible by the right number for the given
    # network depth
    n_div = 2**(n_layers-1)
    if h%n_div != 0:
        h_pad = h+n_div-(h%n_div)
    else:
        h_pad = np.copy(h)
    if w%n_div != 0:
        w_pad = w+n_div-(w%n_div)
    else:
        w_pad = np.copy(w)
    train_img = np.ones((n,h_pad,w_pad))
    train_lbl = np.ones((n,h_pad,w_pad))
    count = 0
    for f in trn_files:
        #-- same file name but different directories for images and labels
        #-- read image and label first
        img = Image.open(os.path.join(trn_subdir,f)).convert('L')
        lbl = Image.open(os.path.join(trn_dir,'labels%s'%crop_str,f.replace('Subset','Front'))).convert('L')
        #-- do permutations with the following:
        #-- 1) flip image Horizontal (spatial)
        #-- 2) flip color (invert)
        #-- 3) ?
        #-- ORIGINAL
        train_img[count][:im_shape[0],:im_shape[1]] = np.array(img)/255.
        train_lbl[count][:im_shape[0],:im_shape[1]] = np.array(lbl)/255.
        count += 1
        if augment:
            #-- INVERT COLORS
            train_img[count][:im_shape[0],:im_shape[1]] = np.array(ImageOps.invert(img))/255.
            train_lbl[count][:im_shape[0],:im_shape[1]] = np.array(lbl)/255.
            count += 1
            #-- MIRROR HORIZONTALLY
            train_img[count][:im_shape[0],:im_shape[1]] = np.array(ImageOps.mirror(img))/255.
            train_lbl[count][:im_shape[0],:im_shape[1]] = np.array(ImageOps.mirror(lbl))/255.
            count += 1

    #-- also get the test data
    n_test = len(tst_files)
    test_img = np.ones((n_test,h_pad,w_pad))
    for i in range(n_test):
        test_img[i][:im_shape[0],:im_shape[1]] = np.array(Image.open(tst_list[i]).convert('L'))/255.

    return {'trn_img':train_img.reshape(n,h_pad,w_pad,1),'trn_lbl':train_lbl.reshape(n,h_pad,w_pad,1),\
        'tst_img':test_img.reshape(n_test,h_pad,w_pad,1),'trn_names':trn_files,'tst_names':tst_files}

def train_model(parameters):
    glacier = parameters['GLACIER_NAME']
    n_batch = int(parameters['BATCHES'])
    n_epochs = int(parameters['EPOCHS'])
    n_layers = int(parameters['LAYERS_DOWN'])
    n_init = int(parameters['N_INIT'])
    suffix = parameters['SUFFIX']
    drop = float(parameters['DROPOUT'])
    if parameters['AUGMENT'] in ['Y','y']:
        augment = True
        aug_str = '_augment'
    else:
        augment = False
        aug_str = ''
    
    if parameters['CROP'] in ['Y','y']:
        crop_str = '_cropped'
    else:
        crop_str = ''

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ddir = os.path.join(current_dir,'%s.dir'%glacier)
    data_dir = os.path.join(ddir, 'data')
    trn_dir = os.path.join(data_dir,'train')
    tst_dir = os.path.join(data_dir,'test')

    #-- set up labels from parameters
    drop_str = ''
    if drop>0:
        drop_str = '_w%.1fdrop'%drop

    #-- load images
    data = load_data(suffix,trn_dir,tst_dir,n_layers,augment,crop_str)

    n,height,width,channels=data['trn_img'].shape
    print('width=%i'%width)
    print('height=%i'%height)

    #-- import mod
    unet = imp.load_source('unet_model', os.path.join(current_dir,'unet_model.py'))
    model,n_tot = unet.unet_model(height=height,width=width,channels=channels,\
        n_init=n_init,n_layers=n_layers,drop=drop)

    #-- checkpoint file
    chk_file = os.path.join(ddir,'unet_model_weights_%ilayers_%iinit%s%s%s%s.h5'\
        %(n_tot,n_init,drop_str,aug_str,suffix,crop_str))

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
            lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
            #-- now fit the model
            model.fit(data['trn_img'], data['trn_lbl'], batch_size=n_batch, epochs=n_epochs, verbose=1,\
                validation_split=0.1, shuffle=True, callbacks=[lr_callback,model_checkpoint])
        else:
            # Compile model (required to make predictions)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #-- if not train model
    else:
        print('Did not find check points. Training model...')
        #-- create checkpoint
        model_checkpoint = keras.callbacks.ModelCheckpoint(chk_file, monitor='loss',\
            verbose=1, save_best_only=True)
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
            verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        #-- now fit the model
        model.fit(data['trn_img'], data['trn_lbl'], batch_size=n_batch, epochs=n_epochs, verbose=1,\
            validation_split=0.1, shuffle=True, callbacks=[lr_callback,model_checkpoint])

    print('Model is trained. Running on test data...')

    #-- make dictionaries for looping through train and test sets
    in_img = {}
    in_img['train'] = data['trn_img']
    in_img['test'] = data['tst_img']
    outdir = {}
    outdir['train'] = trn_dir
    outdir['test'] = tst_dir
    names = {}
    names['train'] = data['trn_names']
    names['test'] = data['tst_names']
    #-- Now test the model on both the test data and the train data
    for t in ['test']:
        out_imgs = model.predict(in_img[t], batch_size=1, verbose=1)
        print out_imgs.shape
        #-- make output directory
        out_subdir = 'output_%ilayers_%iinit%s%s%s%s'\
            %(n_tot,n_init,drop_str,aug_str,suffix,crop_str)
        if (not os.path.isdir(os.path.join(outdir[t],out_subdir))):
            os.mkdir(os.path.join(outdir[t],out_subdir))
        #-- save the test image
        for i in range(len(out_imgs)):
            im = image.array_to_img(out_imgs[i])
            im.save(os.path.join(outdir[t],out_subdir,'%s'%names[t][i].replace('_Subset','')))

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
