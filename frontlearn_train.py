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
import imp

#-- directory setup
#- current directory
ddir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data'))
img_dir = os.path.join(data_dir,'train/images')
lbl_dir = os.path.join(data_dir,'train/labels')
test_dir = os.path.join(data_dir,'train/test')

#-- read in images
def load_train_data()


def main()
    #-- load images
    train_img,train_lbl,test_img = load_train_data()

    #-- import mod
    unet = imp.load_source('unet_model', os.path.join(ddir,'frontlearn_unet.py')
    model = unet()
    #-- create checkpoint
    model_checkpoint = keras.callbacks.ModelCheckpoint('frontlearn_weights.h5', monitor='loss',\
        verbose=1, save_best_only=True)
    #-- now fit the model
    model.fit(train_img, train_lbl, batch_size=5, nb_epoch=10, verbose=1,\
        validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

    #-- Now test the model
    unet_test = model.predict(test_img, batch_size=1, verbose=1)
    np.save(os.path.join(test_dir,'frontlearn_test.npy'),unet_test)
    #-- also save the test image
    out_imgs = np.load(os.path.join(test_dir,'frontlearn_test.npy'))
    for i,m in enumerate(out_imgs):
        im = keras.preprocessing.image.img_to_array(m)
        im.save(os.path.join(test_dir,'%i.jpg'%i))

if __name__ == '__main__':
    main()
