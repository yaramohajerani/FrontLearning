#!/usr/bin/env python
u"""
frontlearn_unet_fixedLayers.py
by Yara Mohajerani (04/2018)

U-net model for glacier front detection;
Note this is just a simple, fixed-layer model
for testing. For trainig use frontlearn_unet_dynamic.py


Update History
        04/2014 Written
"""
import keras.layers as kl
import keras.models as km

def unet_model(height,width,channels):

    #-- define input
    inputs = kl.Input((height,width,channels))

    #-- 1st convlution layer
    #-- c1 has shape (None, h, 2, 20), 20 convlution outputs for 20 filters
    c1 = kl.Conv2D(20,3,activation='relu',padding='same')(inputs)

    #-- pool, 2x2 blockcs
    #-- p1 has shape (None h/2, w/2,  20)
    p1 = kl.MaxPooling2D(pool_size=(2,2))(c1)

    # 2nd convlution layer
    #-- c2 has shape (None, h/2,w/2, 40)
    c2 = kl.Conv2D(40,3,activation='relu',padding='same')(p1)

    #---------------------------------------------
    #-- now go back up to reconsturct the image
    #---------------------------------------------
    #-- concatenate the 1st convolution layer with an upsampled 2nd layer
    #-- where the missing elements in the 2nd layer are padded with 0
    #-- concatenating along the color channels
    #-- upsampled_c2 has shape (None,h,w,40)
    upsampled_c2 = kl.UpSampling2D(size=(2,2))(c2)
    #-- up1 has shape (None, h, w, 60)  (20+40 channels from c2 and c1)
    up1 = kl.concatenate([upsampled_c2,c1],axis=3)
    #-- now do a convlution with the merged upsampled layer
    #-- c4 has shape (None, h, w, 20)
    c3 = kl.Conv2D(20,3,activation='relu',padding='same')(up1)

    #-- convlution across the last 20 filters into 5 channels
    #-- c45has shape (None,h,w,5)
    c4 = kl.Conv2D(5,3,activation='relu',padding='same')(c3)

    #-- do one final sigmoid convolution into just 1 final channel (None,h,w,1)
    c5 = kl.Conv2D(1,1,activation='sigmoid')(c4)

    #-- make model
    model = km.Model(input=inputs,output=c5)
    #-- compile model
    model.compile(loss='binary_crossentropy',optimizer='adam',\
        metrics=['accuracy'])

    #-- return model
    return model
