#!/usr/bin/env python
u"""
sharpen_input.py
by Yara Mohajerani (04/2018)

Do a low-pass or high-pass filter in fourier space to clear up the images
(low pass to experiment with denoising, high pass to sharpen)

Update History
        04/2018 Written
"""
import os
import sys
import getopt
import numpy as np
from glob import glob
from PIL import Image
from scipy import fftpack
import matplotlib.pyplot as plt
from keras.preprocessing import image

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

    train_img = np.zeros((n,h,w,1))
    for i,f in enumerate(trn_files):
        train_img[i,:,:,0] = np.array(Image.open(os.path.join(trn_dir,'images',f)).convert('L'))/255.

    #-- also get the test data
    n_test = len(tst_files)
    #-- get dimensions, force to 1 b/w channel
    w_test,h_test = np.array(Image.open(tst_list[0]).convert('L')).shape
    test_img = np.zeros((n_test,h_test,w_test,1))
    for i in range(n_test):
        test_img[i,:,:,0] = np.array(Image.open(tst_list[i]).convert('L'))/255.

    images = {'train':train_img,'test':test_img}
    names = {'train':trn_files,'test':tst_files}
    return [images,names]

#-- function to read and sharpen data
def sharpen_images(fraction,filter_type):
    #-- first read data
    images,names = load_data()
    #-- make output directory dictionary
    outdir = {}
    outdir['train'] = os.path.join(trn_dir,'images_%s_%.3f'%(filter_type,fraction))
    outdir['test'] = os.path.join(tst_dir,'images_%s_%.3f'%(filter_type,fraction))
    #-- loop through train and test data
    for t in ['train','test']:
        if (not os.path.isdir(outdir[t])):
            os.mkdir(outdir[t])
        #-- loop through images and sharpen them
        for m,n in zip(images[t],names[t]):
            #-- take fft
            rows,cols,channels = m.shape
            f_img = fftpack.fft2(m.reshape(rows,cols))
            #-- make a numpy array copy
            f = f_img.copy()
            #-- get dimensions
            d1,d2 = f.shape

            #-- Set the low frequenies to 0 (to sharpen)
            #-- if you instead want ot get rid of high-frequency noise, set the
            #-- high frequencies to zero
            if filter_type in ['highpass','HighPass','HIGHPASS','high-pass']:
                f[0:int(d1*fraction),:] = 0.
                f[-1*int(d1*fraction):,:] = 0.
                f[:,0:int(d2*fraction)] = 0.
                f[:,-1*int(d2*fraction):] = 0.
            elif filter_type in ['lowpass','LowPass','LOWPASS','low-pass']:
                f[int(d1*fraction):int(d1*(1-fraction)),:] = 0
                f[:,int(d2*fraction):int(d2*(1-fraction))] = 0
            #-- convert back to spatial domain
            im_sharp = fftpack.ifft2(f).real
            #-- convert to 3 dimensions with a color channel
            im_out_array = im_sharp.reshape(rows,cols,1)
            #-- write image to file
            im_out = image.array_to_img(im_out_array)
            im_out.save(os.path.join(outdir[t],'%s'%n))

#-- main function to get user input and call sharpen function
def main():
    #-- Read the system arguments listed after the program
    long_options = ['fraction=','filter=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'F:I:',long_options)

    fraction = 0.1
    filter_type = 'highpass'
    for opt, arg in optlist:
        if opt in ('-F','--fraction'):
            fraction = np.float(arg)
        elif opt in ('-I','--filter'):
            filter_type = arg

    sharpen_images(fraction,filter_type)

if __name__ == '__main__':
    main()
