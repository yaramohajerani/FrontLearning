#!/anaconda2/bin/python2.7
u"""
sw_predict.py
by Yara Mohajerani (09/2018)

This script uses the trained netwrok from 'sw_train.py' to
draw a contiuous boundary for the glacier front

Update History
    09/2018 Written
"""

import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
import imp
import sys
import random
from glob import glob
from PIL import Image
from keras import backend as K
from tensorflow.python.client import device_lib
import tensorflow as tf
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy import ndimage
#-- Print backend information
print 'device information: ', device_lib.list_local_devices()
print 'available GPUs: ', K.tensorflow_backend._get_available_gpus()

#-- read in images
def load_data(suffix,ddir,n_windows_predict,HH,HW):
    #-- make subdirectories for input images
    files = {}
    hbnds = {}
    wbnds = {}
    #-- initialize dicttionaries
    images = {} 
    labels = {}
    for d in ['test','train']:
        subdir = os.path.join(ddir[d],'images%s'%(suffix))
        #-- get a list of the input files
        file_list = glob(os.path.join(subdir,'*.png'))
        #-- get just the file names
        files[d] = [os.path.basename(i) for i in file_list]

        #-- read data
        #-------------------------------------------------------------------------------------------------------------------------
        #-- now make sliding window data
        #-- note first half is centered on boundaries and the second half is not
        #-------------------------------------------------------------------------------------------------------------------------
        #-- take many samples from the input images to predict the boundary
        #-- keep track of which segments belong to each image
        images[d] = {}
        labels[d] = {}
        hbnds[d] = {}
        wbnds[d] = {}
        #-- get indices of boundaery for each image and select from them randomly
        for i,f in enumerate(files[d]):
            images[d][i] = np.zeros((n_windows_predict, 2*HH + 1 , 2*HW + 1))
            hbnds[d][i] = np.zeros((n_windows_predict,2))
            wbnds[d][i] = np.zeros((n_windows_predict,2))
            #-- same file name but different directories for images and labels
            img = np.array(Image.open(os.path.join(subdir,f)).convert('L'))/255.
            labels[d][i] = np.array(Image.open(os.path.join(ddir[d],'labels',\
                f.replace('Subset','Front'))).convert('L'))/255.

            #-- take n_windows_predict random samples from the image
            ih = np.random.randint(HH,high=img.shape[0]-HH,size=n_windows_predict)
            iw = np.random.randint(HW,high=img.shape[1]-HW,size=n_windows_predict)
            for j in range(n_windows_predict):
                images[d][i][j] = img[ih[j]-HH:ih[j]+HH+1,iw[j]-HW:iw[j]+HW+1]
                hbnds[d][i][j] = [ih[j]-HH,ih[j]+HH]
                wbnds[d][i][j] = [iw[j]-HW,iw[j]+HW]

            #-- reshape images for CNN (4D input with channel)
            images[d][i] = images[d][i].reshape(n_windows_predict, 2*HH + 1 , 2*HW + 1, 1)

    #-- assuming all images have the same dimensions, get the size of the input images from the last instance
    img_shape = img.shape

    return [images,labels,hbnds,wbnds,img_shape,files]

def create_model(reg,input_shape,n_init):
    model = Sequential()
    model.add(Conv2D(n_init, (3, 3) ,padding='same',input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(n_init*2, (3, 3) ,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(n_init, kernel_regularizer=l2(reg)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(2, kernel_regularizer=l2(reg)))

    model.add(Activation('softmax'))

    return model

#-- train model on patches of images
def draw_boundary(parameters):
    glacier = parameters['GLACIER_NAME']
    suffix = parameters['SUFFIX']
    HW = np.int(parameters['HALF_WIDTH']) #-- suggested 10
    HH = np.int(parameters['HALF_HEIGHT']) #-- suggested 10
    n_windows = np.int(parameters['N_WINDOWS'])
    n_windows_predict = np.int(parameters['N_WINDOWS_PREDICT'])
    EPOCHS = np.int(parameters['EPOCHS'])
    BATCHES = np.int(parameters['BATCHES'])
    n_init = np.int(parameters['N_INIT'])
    imb_w = np.int(parameters['IMBALANCE_RATIO'])
    reg = np.float(parameters['REGULARIZATION'])

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    glacier_ddir = os.path.join(current_dir,'%s.dir'%glacier)
    data_dir = os.path.join(glacier_ddir, 'data')
    ddir = {}
    ddir['train'] =  os.path.join(data_dir,'train')
    ddir['test'] = os.path.join(data_dir,'test')

    #-- load images
    images,labels,hbnds,wbnds,img_shape,files = load_data(suffix,ddir,n_windows_predict,HH,HW)

     #-- set up model
    model = create_model(reg,(2*HH+1, 2*HW+1,1),n_init)

    #-- set up class weight to deal with imbalance
    if imb_w == 0:
        imb_str = 'auto-'
    else:
        imb_str = str(imb_w)


    #-- checkpoint file
    chk_file = os.path.join(glacier_ddir,'SW_frontlearn_cnn_model_%iHH_%iHW_%inwindows_%iinit_%simbalance_%.1e%s.h5'\
        %(HH,HW,n_windows,n_init,imb_str,reg,suffix))

    #-- if file exists, just read model from file
    if os.path.isfile(chk_file):
        print('Check point exists; loading model from file.')
        # load weights
        model.load_weights(chk_file)
        # Compile model
        model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    #-- if not train model
    else:
        sys.exit('Did not find check points. Need to training model first.')

    #-- make predictions
    for d in ['test','train']:
        #-- loop through images
        for i in range(len(images[d])):
            pred = model.predict(images[d][i])

            #-- get the boxes that have boundaries
            ind = np.nonzero(pred[:,1] > 0.5)

            #-- plot the center of each box that has a boundary
            hcntr = np.mean(hbnds[d][i][ind],axis=1)
            wcntr = np.mean(wbnds[d][i][ind],axis=1)

            ypts1 = []
            xpts1 = []

            #-- get the vertical mean and std of all the points
            y_avg = np.mean(hcntr)
            y_std = np.std(hcntr)
            x_avg = np.mean(wcntr)
            x_std = np.std(wcntr)
            #-- scaling for sigma
            yscale = 2.
            xscale = 2.
            pts = []
            n_pts = len(wcntr)
            for p in range(n_pts):
                if (int(hcntr[p]) > (y_avg - y_std*yscale) and int(hcntr[p]) < (y_avg + y_std*yscale) and \
                    int(wcntr[p]) > (x_avg - x_std*xscale) and int(wcntr[p]) < (x_avg + x_std*xscale)):
                    xpts1.append(int(wcntr[p]))
                    ypts1.append(int(hcntr[p]))

            xpts1 = np.array(xpts1)
            ypts1 = np.array(ypts1)

            #-- remove outliers one more time
            y_avg = np.mean(ypts1)
            y_std = np.std(ypts1)
            x_avg = np.mean(xpts1)
            x_std = np.std(xpts1)
            #-- scaling for sigma
            yscale = 3.0
            xscale = 2.0 #1.5
            ypts2 = []
            xpts2 = []
            n_pts = len(xpts1)
            for p in range(n_pts):
                if (ypts1[p] > (y_avg - y_std*yscale) and ypts1[p] < (y_avg + y_std*yscale) and \
                    xpts1[p] > (x_avg - x_std*xscale) and xpts1[p] < (x_avg + x_std*xscale)):
                    xpts2.append(xpts1[p])
                    ypts2.append(ypts1[p])
                    pts.append(Point(xpts1[p],ypts1[p]))
            frontline = LineString(pts)
            xpts2 = np.array(xpts2)
            ypts2 = np.array(ypts2)

            #-- apply gaussian smoothing filter
            ind_sort = np.argsort(xpts2)
            print(x_std)
            x = ndimage.gaussian_filter1d(xpts2[ind_sort], x_std/2.)
            y = ndimage.gaussian_filter1d(ypts2[ind_sort], y_std/2.)

            fig, ax = plt.subplots(figsize=(7,4))
            #-- plot target label
            ax.imshow(labels[d][i],alpha=0.8,cmap='gray',zorder=1)
            #-- fake line for legend
            ax.plot([],[],color='black',label='True Front')
            #-- plot raw poitns
            ax.scatter(wcntr,hcntr,s=0.5,alpha=0.8,color='r',label='NN output',zorder=2)
            #-- plot horizontal area of consideration
            #ax.plot(np.arange(w),np.ones(w)*y_avg,'g',zorder=3)
            ax.fill_between(np.arange(img_shape[1]),np.ones(img_shape[1])*(y_avg - y_std*yscale),\
                y2=np.ones(img_shape[1])*(y_avg + y_std*yscale),alpha=0.2,color='g',label='Horizontal Filter',zorder=3)
            #-- plot vertical area of consideration
            #ax.plot(np.ones(h)*x_avg,np.arange(h),'b-')
            ax.axvspan(x_avg-x_std*xscale,x_avg+x_std*xscale,alpha=0.2,color='b',label='Vertical Filter',zorder=4)
            #-- plot gnereated front
            ax.plot(x,y,color='cyan',linewidth=2,label='Generated Front',zorder=5)
            ax.legend(bbox_to_anchor=(1., 1.))
            plt.subplots_adjust(left=-0.1)
            #plt.show()
            #-- save
            subdir = os.path.join(ddir[d],'output_SW_cnn_%iHH_%iHW_%inwindows_%iinit_%simbalance_%.1e%s'\
                %(HH,HW,n_windows,n_init,imb_str,reg,suffix))
            if not os.path.isdir(subdir):
                os.mkdir(subdir)
            plt.savefig(os.path.join(subdir,'%s_postprocess.png'%(files[d][i][:-4])))
            plt.close(fig)
            
            #-- alternate approach: get the average boundary position by doing a weighted average of all the boxes
            #-- since 1 is for boundary and 0 is for no boundary. And this way we take in-between probabilities
            #-- into account.


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
            draw_boundary(parameters)

if __name__ == '__main__':
    main()