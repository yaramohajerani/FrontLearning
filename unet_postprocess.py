#!/usr/bin/env python
u"""
frontlearn_postprocess.py
by Yara Mohajerani

Post-Processing of the predictions of the neural network

TO BE COMPLETED

History
    04/2018 Written
"""
import os
import numpy as np
import imp
import sys
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy import ndimage

def post_process(parameters):
    glacier = parameters['GLACIER_NAME']
    n_batch = int(parameters['BATCHES'])
    n_epochs = int(parameters['EPOCHS'])
    n_layers = int(parameters['LAYERS_DOWN'])
    n_init = int(parameters['N_INIT'])
    suffix = parameters['SUFFIX']
    drop = float(parameters['DROPOUT'])

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.path.join(current_dir,'..','FrontLearning_data')
    glacier_ddir = os.path.join(main_dir,'%s.dir'%glacier)
    data_dir = os.path.join(glacier_ddir, 'data')
    trn_dir = os.path.join(data_dir,'train')
    tst_dir = os.path.join(data_dir,'test')

    #-- set up labels from parameters
    drop_str = ''
    if drop>0:
        drop_str = '_w%.1fdrop'%drop

    #-- total number of layers
    layers_tot = 2*n_layers+1

    #-- read in output data of the neural network
    subdir = os.path.join(tst_dir,'output_%ibtch_%iepochs_%ilayers_%iinit%s%s'\
        %(n_batch,n_epochs,layers_tot,n_init,drop_str,suffix))

    #-- get a list of the input files
    in_list = sorted(glob(os.path.join(subdir,'*png')))
    in_list = sorted([fn for fn in glob(os.path.join(subdir,'*png'))
         if not os.path.basename(fn).endswith('threshold.png')])
    n_files = len(in_list)
    filenames = [os.path.basename(i) for i in in_list]
    print(filenames)
    h,w = np.array(Image.open(in_list[0]).convert('L')).shape

    #-- also get the labels for plotting
    lbl_list = sorted(glob(os.path.join(tst_dir,'labels/*.png')))

    #-- vectorize files
    for i in range(n_files):
        img = np.array(Image.open(in_list[i]).convert('L'))/255.
        lbl = np.array(Image.open(lbl_list[i]).convert('L'))/255.

        #-- set a threshold for points that are to be identified as the front
        at = 0.8 #-- amplitude threshold
        img_flat = img.flatten()
        ind_black = np.squeeze(np.nonzero(img_flat <= at))
        ind_white = np.squeeze(np.nonzero(img_flat > at))
        img_flat[ind_black] = 0.
        img_flat[ind_white] = 1.
        img2 = img_flat.reshape(img.shape)

        #-- now draw a line through the points
        #-- first get the index of all the black points
        ind_2D = np.squeeze(np.nonzero(img2 == 0.))

        ypts1 = [] #ind_2D[0,:]
        xpts1 = [] #ind_2D[1,:]

        #-- get the vertical mean and std of all the points
        y_avg = np.mean(ind_2D[0,:])
        y_std = np.std(ind_2D[0,:])
        x_avg = np.mean(ind_2D[1,:])
        x_std = np.std(ind_2D[1,:])
        #-- scaling for sigma
        yscale = 2. #1.
        xscale = 2.
        pts = []
        n_pts = len(ind_2D[1,:])
        for p in range(n_pts):
            if (int(ind_2D[0,p]) > (y_avg - y_std*yscale) and int(ind_2D[0,p]) < (y_avg + y_std*yscale) and \
                int(ind_2D[1,p]) > (x_avg - x_std*xscale) and int(ind_2D[1,p]) < (x_avg + x_std*xscale)):
                xpts1.append(int(ind_2D[1,p]))
                ypts1.append(int(ind_2D[0,p]))

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

        #krnl = KernelReg(xpts,ypts,'u')
        """
        #-- make an x array to be plotted
        x = np.unique(xpts2)
        y = np.zeros(len(x))    #krnl.fit(x)[0] # first term is the actual curve

        for j,ix in enumerate(x):
            ind_ix = np.squeeze(np.nonzero(xpts2==ix))
            y[j] = np.mean(ypts2[ind_ix])
        """
        #-- apply gaussian smoothing filter
        ind_sort = np.argsort(xpts2)
        print(x_std)
        x = ndimage.gaussian_filter1d(xpts2[ind_sort], x_std/2.)
        y = ndimage.gaussian_filter1d(ypts2[ind_sort], y_std/2.)

        fig, ax = plt.subplots(figsize=(7,4))
        #-- plot target label
        ax.imshow(lbl,alpha=0.8,cmap='gray',zorder=1)
        #-- fake line for legend
        ax.plot([],[],color='black',label='True Front')
        #-- plot raw poitns
        ax.scatter(ind_2D[1,:],ind_2D[0,:],s=0.5,alpha=0.8,color='r',label='NN output',zorder=2)
        #-- plot horizontal area of consideration
        #ax.plot(np.arange(w),np.ones(w)*y_avg,'g',zorder=3)
        ax.fill_between(np.arange(w),np.ones(w)*(y_avg - y_std*yscale),\
            y2=np.ones(w)*(y_avg + y_std*yscale),alpha=0.2,color='g',label='Horizontal Filter',zorder=3)
        #-- plot vertical area of consideration
        #ax.plot(np.ones(h)*x_avg,np.arange(h),'b-')
        ax.axvspan(x_avg-x_std*xscale,x_avg+x_std*xscale,alpha=0.2,color='b',label='Vertical Filter',zorder=4)
        #-- plot gnereated front
        ax.plot(x,y,color='cyan',linewidth=2,label='Generated Front',zorder=5)
        ax.legend(bbox_to_anchor=(1., 1.))
        plt.subplots_adjust(left=-0.1)
        #plt.show()
        plt.savefig(os.path.join(subdir,'%s_postprocess_%.2fthreshold.png'%(filenames[i][:-4],at)))
        plt.close(fig)

#-- main function to get parameters and pass them along to the postprocessing function
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
            post_process(parameters)

if __name__ == '__main__':
    main()
