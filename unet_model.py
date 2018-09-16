#!/usr/bin/env python
u"""
frontlearn_unet_dynamic.py
by Yara Mohajerani (Last Update 09/2018)

Construct a dynamic u-net model with a variable
number of layers for glacier calving front detection.

Update History
        09/2018 Add custom loss functin with weights for 
                black and white pixels
        04/2018 Written
"""
from keras import backend as K
import keras.layers as kl
import keras.models as km
import copy

def weighted_binary_crossentropy(weights):
    """
    note the weighted binary cross entopy loss would be
     alpha.ytrue.log(ypred) + beta.(1 - ytrue)*log(1 - ypred)

    so alpha would be the weight of getting white pixels wrong (ytrue=1)
    and beta is the weight of getting black pixels  wrong (ytue=0)

    since we don't want ot miss any boundary points, we want to increase
    beta in this case.
    """
    w = K.variable(weights)

    def loss(y_true, y_pred):
        if y_true == 1:
            return -K.log(y_pred)*w[0]
        else:
            return -K.log(1 - y_pred)*w[1]
    
    return loss

def unet_model(height=0,width=0,channels=1,n_init=12,n_layers=2,drop=0,weights=[1,1]):

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
        #-- now do a convlution with the merged upsampled layer
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
    c[i] = kl.Conv2D(3,3,activation='relu',padding='same')(c[i-1])
    #-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
    i += 1
    c[i] = kl.Conv2D(1,1,activation='sigmoid')(c[i-1])
    print('Total Number of layers: ',i)

    #-- make model
    model = km.Model(input=inputs,output=c[i])
    #-- compile model
    loss = weighted_binary_crossentropy(weights)
    model.compile(loss=loss,optimizer='adam', metrics=['accuracy'])

    #-- return model
    return [model,i]
