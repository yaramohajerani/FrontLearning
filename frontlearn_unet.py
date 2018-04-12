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

def unet_model(height,width,channels):

    #-- define input
    inputs = kl.Input((height,width,channels))

    #-- 1st convlution layer
    #-- c1 has shape (None, h, 2, 12), 12 convlution outputs for 12 filters
    c1 = kl.Conv2D(12,3,activation='relu',padding='same')(inputs)

    #-- pool, 2x2 blockcs
    #-- p1 has shape (None h/2, w/2,  12)
    p1 = kl.MaxPooling2D(pool_size=(2,2))(c1)

    # 2nd convlution layer
    #-- c2 has shape (None, h/2,w/2, 24)
    c2 = kl.Conv2D(24,3,activation='relu',padding='same')(p1)

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    #-- concatenate the 1st convolution layer with an upsampled 2nd layer
    #-- where the missing elements in the 2nd layer are padded with 0
    #-- concatenating along the color channels
    #-- upsampled_c2 has shape (None,h,w,24)
    upsampled_c2 = kl.UpSampling2D(size=(2,2))(c2)
    #-- up1 has shape (None, h, w, 36)  (12+24 channels from c2 and c1)
    up1 = kl.concatenate([upsampled_c2,c1],axis=3)
    #-- now do a convlution with the merged upsampled layer
    #-- c4 has shape (None, h, w, 12)
    c4 = kl.Conv2D(12,3,activation='relu',padding='same')(up1)

    #-- convlution across the last 12 filters into 3 channels
    #-- c45has shape (None,h,w,3)
    c5 = kl.Conv2D(3,3,activation='relu',padding='same')(c4)

    #-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
    c6 = kl.Conv2D(1,1,activation='sigmoid')(c5)

    #-- make model
    model = km.Model(input=inputs,output=c6)
    #-- compile model
    model.compile(loss='binary_crossentropy',optimizer='adam',\
        metrics=['accuracy'])

    #-- return model
    return model
