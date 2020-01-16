#!/anaconda2/bin/python2.7
u"""
frontlearn_unet.py
by Yara Mohajerani (Last Update 01/2019)

Construct a dynamic u-net model with a variable
number of layers for glacier calving front detection.

Update History
    01/2019 Fix batch normalization axis input
    09/2018 Add multiple functions to test different versions
            Don't compile (compile in main script to allow for 
            different weighting experiments)
            Add multiple functions with different architectures
            Add new option for batch normalization instead of dropout
    04/2018 Written
"""
from keras import backend as K
import keras.layers as kl
import keras.models as km
import copy
import sys
import keras
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
import os
import imp

current_dir = os.path.dirname(os.path.realpath(__file__))
BilinearUpsampling = imp.load_source('BilinearUpsampling', os.path.join(current_dir,'..','keras-deeplab-v3-plus','model_cfm_dual_wide_x65.py'))

#---------------------------------------------------------------------------------------
#-- linearly scale the size of each convolution layer (i.e. initial*i for the ith layer)
#---------------------------------------------------------------------------------------
def unet_model_linear_dropout(height=0,width=0,channels=1,n_init=12,n_layers=2,drop=0):

    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    for i in range(1,n_layers+1):
        #-- convolution layer
        c[i] = kl.Conv2D(n_init*i,3,activation='relu',padding='same')(p[i-1])
        if drop != 0:
            c[i] = kl.Dropout(drop)(c[i])
        c[i] = kl.Conv2D(n_init*i,3,activation='relu',padding='same')(c[i])

        #-- pool, 2x2 blockcs
        #-- don't do pooling for the last down layer
        if i != n_layers:
            p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
        count += 1

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    print('Max Number of Convlution Filters: ',n_init*i)
    while count>1:
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[count])
        up[i] = kl.concatenate([upsampled_c[i],c[count-1]],axis=3)
        #-- now do a convolution with the merged upsampled layer
        i += 1
        c[i] = kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(up[i-1])
        if drop != 0:
            c[i] = kl.Dropout(drop)(c[i])
        c[i] = kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(c[i])
        #-- counter decreases as we go back up
        count -= 1

    print('Number of Convlution Filters at the end of up segment: ',n_init*count)
    #-- convlution across the last n_iniy filters into 3 channels
    i += 1
    c[i] = kl.Conv2D(2,3,activation='sigmoid',padding='same')(c[i-1])
    print('output shape: ', c[i].shape)
    print('Total Number of layers: ',i)

    #-- make model
    model = km.Model(input=inputs,output=c[i])

    #-- return model
    return model



#-----------------------------------------------------------------------------------
#-- double the size of each convolution layer 
#-----------------------------------------------------------------------------------
def unet_model_double_dropout(height=0,width=0,channels=1,n_init=12,n_layers=2,drop=0):

    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    n_filts = copy.copy(n_init)
    for i in range(1,n_layers+1):
        #-- convolution layer
        c[i] = kl.Conv2D(n_filts,3,activation='relu',padding='same')(p[i-1])
        if drop != 0:
            c[i] = kl.Dropout(drop)(c[i])
        c[i] = kl.Conv2D(n_filts,3,activation='relu',padding='same')(c[i])

        #-- pool, 2x2 blockcs
        #-- don't do pooling for the last down layer
        #-- also don't double the filter numbers
        if i != n_layers:
            p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
            n_filts *= 2
        count += 1


    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    print('Max Number of Convlution Filters: ',n_filts)
    while count>1:
        n_filts = int(n_filts/2)
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[count])
        # upsampled_c[i] = BilinearUpsampling.BilinearUpsampling(upsampling=(2,2))(c[count])
        up[i] = kl.concatenate([upsampled_c[i],c[count-1]],axis=3)
        #-- now do a convlution with the merged upsampled layer
        i += 1
        c[i] = kl.Conv2D(n_filts,3,activation='relu',padding='same')(up[i-1])
        if drop != 0:
            c[i] = kl.Dropout(drop)(c[i])
        c[i] = kl.Conv2D(n_filts,3,activation='relu',padding='same')(c[i])
        #-- counter decreases as we go back up
        count -= 1

    print('Number of Convlution Filters at the end of up segment: ',n_filts)
    #-- convlution across the last n_iniy filters into 3 channels
    i += 1
    c[i] = kl.Conv2D(2,3,activation='sigmoid',padding='same')(c[i-1])
    print('output shape: ', c[i].shape)
    print('Total Number of layers: ',i)


    #-- make model
    model = km.Model(input=inputs,output=c[i])

    #-- return model
    return model




#-----------------------------------------------------------------------------------
#-- batch normalization instread of dropout for "linear" architecture
#-----------------------------------------------------------------------------------
def unet_model_linear_normalized(height=0,width=0,channels=1,n_init=12,n_layers=2):

    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    for i in range(1,n_layers+1):
        #-- convolution layer
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_init*i,3,activation='relu',padding='same')(p[i-1]))
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_init*i,3,activation='relu',padding='same')(c[i]))

        #-- pool, 2x2 blockcs
        #-- don't do pooling for the last down layer
        if i != n_layers:
            p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
        count += 1

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    print('Max Number of Convlution Filters: ',n_init*i)
    while count>1:
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[count])
        up[i] = kl.concatenate([upsampled_c[i],c[count-1]],axis=3)
        #-- now do a convolution with the merged upsampled layer
        i += 1
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(up[i-1]))
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(c[i]))
        #-- counter decreases as we go back up
        count -= 1

    print('Number of Convlution Filters at the end of up segment: ',n_init*count)
    #-- convlution across the last n_iniy filters into 3 channels
    i += 1
    c[i] = BatchNormalization(axis=-1)(kl.Conv2D(2,3,activation='sigmoid',padding='same')(c[i-1]))
    print('output shape: ', c[i].shape)
    print('Total Number of layers: ',i)

    #-- make model
    model = km.Model(input=inputs,output=c[i])

    #-- return model
    return model




#-----------------------------------------------------------------------------------
#-- batch normalization instread of dropout for "double" architecture
#-----------------------------------------------------------------------------------
def unet_model_double_normalized(height=0,width=0,channels=1,n_init=12,n_layers=2):
    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    n_filts = copy.copy(n_init)
    for i in range(1,n_layers+1):
        #-- convolution layer
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_filts,3,activation='relu',padding='same')(p[i-1]))
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_filts,3,activation='relu',padding='same')(c[i]))

        #-- pool, 2x2 blockcs
        #-- don't do pooling for the last down layer
        #-- also don't double the filter numbers
        if i != n_layers:
            p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
            n_filts *= 2
        count += 1

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    print('Max Number of Convlution Filters: ',n_filts)
    while count>1:
        n_filts /= 2
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[count])
        up[i] = kl.concatenate([upsampled_c[i],c[count-1]],axis=3)
        #-- now do a convlution with the merged upsampled layer
        i += 1
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_filts,3,activation='relu',padding='same')(up[i-1]))
        c[i] = BatchNormalization(axis=-1)(kl.Conv2D(n_filts,3,activation='relu',padding='same')(c[i]))
        #-- counter decreases as we go back up
        count -= 1

    print('Number of Convlution Filters at the end of up segment: ',n_filts)
    #-- convlution across the last n_iniy filters into 3 channels
    i += 1
    c[i] = BatchNormalization(axis=-1)(kl.Conv2D(2,3,activation='sigmoid',padding='same')(c[i-1]))
    print('output shape: ', c[i].shape)
    print('Total Number of layers: ',i)


    #-- make model
    model = km.Model(input=inputs,output=c[i])

    #-- return model
    return model
