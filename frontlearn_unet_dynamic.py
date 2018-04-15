#!/usr/bin/env python
u"""
frontlearn_unet_dynamic.py
by Yara Mohajerani (04/2018)

Construct a dynamic u-net model with a variable
number of layers for glacier front detection.

To be added: dropout

Update History
        04/2014 Written
"""
import keras.layers as kl
import keras.models as km

def unet_model(height=0,width=0,channels=1,n_init=12,n_layers=2,drop=0):

    #-- define input
    inputs = kl.Input((height,width,channels))

    c = {}
    p = {}
    count = 0
    #-- define input
    p[0] = inputs
    for i in range(1,n_layers+1):
        #-- convlution layer
        c[i] = kl.Conv2D(n_init*i,3,activation='relu',padding='same')(p[i-1])
        #-- pool, 2x2 blockcs
        p[i] = kl.MaxPooling2D(pool_size=(2,2))(c[i])
        count += 1

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    upsampled_c = {}
    up = {}
    while count>1:
        #-- concatenate the 1st convolution layer with an upsampled 2nd layer
        #-- where the missing elements in the 2nd layer are padded with 0
        #-- concatenating along the color channels
        upsampled_c[i] = kl.UpSampling2D(size=(2,2))(c[count])
        up[i] = kl.concatenate([upsampled_c[i],c[count-1]],axis=3)
        #-- now do a convlution with the merged upsampled layer
        i += 1
        c[i] = kl.Conv2D(n_init*(count-1),3,activation='relu',padding='same')(up[i-1])
        #-- counter decreases as we go back up
        count -= 1

    #-- convlution across the last n_iniy filters into 3 channels
    i += 1
    c[i] = kl.Conv2D(3,3,activation='relu',padding='same')(c[i-1])
    #-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
    i += 1
    c[i] = kl.Conv2D(1,1,activation='sigmoid')(c[i-1])

    #-- make model
    model = km.Model(input=inputs,output=c[i])
    #-- compile model
    model.compile(loss='binary_crossentropy',optimizer='adam',\
        metrics=['accuracy'])

    #-- return model
    return [model,i]
