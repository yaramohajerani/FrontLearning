#!/usr/bin/env python
u"""
frontlearn_unet.py
by Yara Mohajerani (04/2018)

Construct a u-net model for glacier grounding line
detection

Update History
        04/2014 Written
"""
import keras.layers as kl
import keras.models as km

def unet_model(channels,height,width):

    #-- define input
    inputs = kl.Input(channels,height,width)

    #-- 1st convlution layer
    c1 = kl.conv2D(12,3,3,activation='relu',padding='valid')(inputs)
    print 'c1 shape: ', c1.shape
    #-- pool, 2x2 blockcs
    p1 = kl.MaxPooling2D(pool_size=(2,2))(c1)
    print 'p1 shaoe: ', p1.shape

    # 2nd convlution layer
    c2 = kl.conv2D(24,3,3,activation='relu',padding='valid')(p1)
    print 'c2 shape: ', c2.shape

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    #-- concatenate the 1st convolution layer with an upsampled 2nd layer
    #-- where the missing elements in the 2nd layer are padded with 0
    #-- concatenating along the color channels
    up1 = kl.concatenate(axis=1)([kl.UpSampling2D(size=(2,2),c2),c1])
    #-- now do a convlution with the merged upsampled layer
    c4 = kl.conv2D(12,3,3,activation='relu',padding='valid')(up1)

    #-- convlution across the last 12 filters into 3 channels
    c5 = kl.conv2D(3,3,3,activation='relu',padding='valid')(c4)

    #-- do one final sigmoid convolution into just 1 final channel
    c6 = kl.conv2D(1,1,1,activation='sigmoid')(c5)

    #-- make model
    model = km.Model(input=inputs,output=c6)
    #-- compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',\
        metrics=['accuracy'])

    #-- return model
    return model
