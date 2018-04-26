#!/usr/bin/env python
u"""
frontlearn_preprocess.py
by Yara Mohajerani (04/2018)

Pre-process input images to improve learning

Update History
        04/2018 Written
"""
import os
import sys
import getopt
import numpy as np
from glob import glob
from PIL import Image,ImageEnhance

#-- directory setup
#- current directory
ddir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data'))
trn_dir = os.path.join(data_dir,'train')
tst_dir = os.path.join(data_dir,'test')

#-- read in images
def load_data():
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

    train_img = []
    for i,f in enumerate(trn_files):
        train_img.append(Image.open(os.path.join(trn_dir,'images',f)))

    #-- also get the test data
    n_test = len(tst_files)
    #-- get dimensions, force to 1 b/w channel
    w_test,h_test = np.array(Image.open(tst_list[0]).convert('L')).shape
    test_img = []
    for i in range(n_test):
        test_img.append(Image.open(tst_list[i]))

    images = {'train':train_img,'test':test_img}
    names = {'train':trn_files,'test':tst_files}
    return [images,names]

#-- function to read and sharpen data
def enhance_images(sharpness,contrast):
    #-- first read data
    images,names = load_data()
    #-- make output directory dictionary
    outdir = {}
    outdir['train'] = os.path.join(trn_dir,'images_sharpness%.1f_contrast%.1f'%(sharpness,contrast))
    outdir['test'] = os.path.join(tst_dir,'images_sharpness%.1f_contrast%.1f'%(sharpness,contrast))
    #-- loop through train and test data
    for t in ['train','test']:
        if (not os.path.isdir(outdir[t])):
            os.mkdir(outdir[t])
        #-- loop through images and ehnance
        for m,n in zip(images[t],names[t]):
            #-- first blur the images to get rid of all the noise
            sharp_obj = ImageEnhance.Sharpness(m)
            blurred = sharp_obj.enhance(sharpness)
            contr_obj = ImageEnhance.Contrast(blurred)
            final = contr_obj.enhance(contrast)

            #-- write image to file
            final.save(os.path.join(outdir[t],'%s'%n))

#-- main function to get user input and call sharpen function
def main():
    #-- Read the system arguments listed after the program
    long_options = ['sharpness=','contrast=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'=S:C:',long_options)

    sharpness= 0.1
    contrast = 4
    for opt, arg in optlist:
        if opt in ('-S','--sharpness'):
            sharpness = np.float(arg)
        elif opt in ('-C','--contrast'):
            contrast = np.float(arg)

    enhance_images(sharpness,contrast)

if __name__ == '__main__':
    main()
