#!/anaconda2/bin/python2.7
u"""
frontlearn_postprocess.py
by Yara Mohajerani

Post-Processing of the predictions of the neural network

History
    09/2018 Updated to break the output of "crop_output.py" into line
            segments and draw the smooth line through the longest line
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
import sklearn.neighbors

def post_process(parameters):
    glacier = parameters['GLACIER_NAME']
    n_layers = int(parameters['LAYERS_DOWN'])
    n_init = int(parameters['N_INIT'])
    suffix = parameters['SUFFIX']
    drop = float(parameters['DROPOUT'])
    imb_str = '_%.2fweight'%(float(parameters['imb_str']))
    at = float(parameters['THRESHOLD'])
    if at != 0:
        threshold_str = '%.2fthreshold'%at
    else:
        threshold_str = 'nothreshold'
    #-- set up configurations based on parameters
    if parameters['AUGMENT'] in ['Y','y']:
        aug_str = '_augment'
    else:
        aug_str = ''
    
    if parameters['CROP'] in ['Y','y']:
        crop_str = '_cropped'
    else:
        crop_str = ''
    
    if parameters['NORMALIZE'] in ['y','Y']:
        norm_str = '_normalized'
    else:
        norm_str = ''
    
    if parameters['LINEAR'] in ['Y','Y']:
        lin_str = '_linear'
    else:
        lin_str = ''

    drop_str = ''
    if drop>0:
        drop_str = '_w%.1fdrop'%drop

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.path.join(current_dir,'..','FrontLearning_data')
    glacier_ddir = os.path.join(main_dir,'%s.dir'%glacier)
    data_dir = os.path.join(glacier_ddir, 'data')
    trn_dir = os.path.join(data_dir,'train')
    tst_dir = os.path.join(data_dir,'test')
    
    #-- read in output data of the neural network
    subdir = os.path.join(tst_dir,'output_%ilayers_%iinit%s%s%s%s%s%s%s'\
            %(n_layers,n_init,lin_str,imb_str,drop_str,norm_str,aug_str,suffix,crop_str))

    #-- get a list of the input files
    in_list = sorted([fn for fn in glob(os.path.join(subdir,'*%s.png'%threshold_str))
         if not os.path.basename(fn).endswith('postprocess.png')])
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

        #-- now draw a line through the points
        #-- first get the index of all the black points
        ind_2D = np.squeeze(np.nonzero(img == 0.))

        ypts1 = []
        xpts1 = []

        #-- get the vertical mean and std of all the points
        y_avg = np.mean(ind_2D[0,:])
        y_std = np.std(ind_2D[0,:])
        x_avg = np.mean(ind_2D[1,:])
        x_std = np.std(ind_2D[1,:])
        #-- scaling for sigma
        yscale = 5.
        xscale = 5.
        n_pts = len(ind_2D[1,:])
        for p in range(n_pts):
            if (int(ind_2D[0,p]) > (y_avg - y_std*yscale) and int(ind_2D[0,p]) < (y_avg + y_std*yscale) and \
                int(ind_2D[1,p]) > (x_avg - x_std*xscale) and int(ind_2D[1,p]) < (x_avg + x_std*xscale)):
                xpts1.append(int(ind_2D[1,p]))
                ypts1.append(int(ind_2D[0,p]))

        xpts1 = np.array(xpts1)
        ypts1 = np.array(ypts1)

        #-- break down image into individual lines by getting indices that are no more than 1 pixel away
        #-- from each other. Note 'front' is a boolean array (True / 1 for boundary)
        #-- note that the indices are not in order of radial distance so first we need to sort the points one 
        #-- by one based on distance
        indices = np.vstack((ypts1,xpts1)).transpose() #ind_2D.transpose() # dimensions = num_pts x 2
        print indices.shape
        #-- counter for segments
        s = 0 
        #-- start with the 0 point and add it to the ordered list and then delete
        ind = 0
        #-- dictionary for indices of segments. Add first point to segment 0
        seg = {}
        seg[s] = [indices[ind,:]]
        indices = np.delete(indices,ind,0)
        #-- loop through points until there are no more points left
        while len(indices) > 0:
            #-- use Ball Tree to search for nearest point to the last point of the current segment
            tree = sklearn.neighbors.BallTree(indices, metric='euclidean')
            dist,ind = tree.query(seg[s][-1].reshape(1,2), k=1)
            #-- assign new point and delete it
            if np.squeeze(dist) <=  15.:
                seg[s].append(indices[ind])
            else:
                s += 1
                seg[s] = [indices[ind]]
            indices = np.delete(indices,ind,0)
            

        #-- now plot the longest segment
        lens = [len(seg[k]) for k in seg.keys()]
        #ind_sorted = np.argsort(lens)
        #max_ind = ind_sorted[-2] 
        max_ind = np.argmax(lens)
        new_im = np.zeros(img.shape)
        for pix_count in range(lens[max_ind]):
            new_im[np.squeeze(np.squeeze(seg[max_ind])[pix_count])[0],\
                np.squeeze(np.squeeze(seg[max_ind])[pix_count])[1]] = 1.

        #-- apply gaussian smoothing filter
        xpts2,ypts2 = np.squeeze(np.nonzero(new_im.transpose()))
        ind_sort = np.argsort(xpts2)
        x = ndimage.gaussian_filter1d(xpts2[ind_sort], np.std(xpts2)/2.)
        y = ndimage.gaussian_filter1d(ypts2[ind_sort], np.std(ypts2)/2.)

        fig, ax = plt.subplots(figsize=(7,4))
        #-- plot target label
        ax.imshow(lbl,alpha=0.8,cmap='gray',zorder=4)
        #ax.imshow(new_im,alpha=0.8,cmap='gray',zorder=5)
        #-- fake line for legend
        ax.plot([],[],color='black',label='True Front')
        #-- plot raw poitns
        ax.scatter(ind_2D[1,:],ind_2D[0,:],s=0.5,alpha=0.8,color='r',label='NN output',zorder=3)
        #-- plot horizontal area of consideration
        #ax.plot(np.arange(w),np.ones(w)*y_avg,'g',zorder=3)
        #ax.fill_between(np.arange(w),np.ones(w)*(y_avg - y_std*yscale),\
        #    y2=np.ones(w)*(y_avg + y_std*yscale),alpha=0.2,color='g',label='Horizontal Filter',zorder=1)
        #-- plot vertical area of consideration
        #ax.plot(np.ones(h)*x_avg,np.arange(h),'b-')
        #ax.axvspan(x_avg-x_std*xscale,x_avg+x_std*xscale,alpha=0.2,color='b',label='Vertical Filter',zorder=2)
        #-- plot gnereated front
        ax.plot(x,y,color='cyan',linewidth=2,label='Generated Front',zorder=5)
        ax.legend(bbox_to_anchor=(1., 1.))
        plt.subplots_adjust(left=-0.1)
        #plt.show()
        plt.savefig(os.path.join(subdir,'%s_%s_postprocess.png'%(filenames[i][:-4],threshold_str)))
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
