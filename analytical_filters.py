#!/anaconda2/bin/python2.7
u"""
analytical_filters.py
by Yara Mohajerani (Last Update 09/2018)

Use Canny edge detector from sckikit-image to detect edges

Update History
        09/2018 written
"""
import os
import sys
import numpy as np
from glob import glob
from PIL import Image, ImageFilter
from skimage import feature
import matplotlib.pyplot as plt
import scipy.misc
from skimage.morphology import skeletonize
from skimage.future import graph
from skimage import data, segmentation, color, filters, io
import sklearn.neighbors
from skimage.filters import sobel
from scipy import ndimage

PLOT = False

#-- read in images
def load_data(suffix,ddir):
    #-- initialize dicttionaries
    images = {} 
    files = {}
    for d in ['test']:#,'train']:
        subdir = os.path.join(ddir[d],'images%s'%(suffix))
        #-- get a list of the input files
        file_list = glob(os.path.join(subdir,'*.png'))
        #-- get just the file names
        files[d] = [os.path.basename(i) for i in file_list]
    
        #-- read data
        images[d] = {}
        #-- get indices of boundaery for each image and select from them randomly
        for i,f in enumerate(files[d]):
            #-- same file name but different directories for images and labels
            images[d][i] = np.array(Image.open(os.path.join(subdir,f)).convert('L'))/255.

    return [images,files]

#-- train model and make predictions
def run_filter(parameters):
    glacier = parameters['GLACIER_NAME']
    suffix = parameters['SUFFIX']
    filter = parameters['FILTER']
    threshold = np.float(parameters['THRESHOLD'])

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.path.join(current_dir,'..','FrontLearning_data')
    glacier_ddir = os.path.join(main_dir,'%s.dir'%glacier)
    data_dir = os.path.join(glacier_ddir, 'data')
    ddir = {}
    ddir['train'] = os.path.join(data_dir,'train')
    ddir['test'] = os.path.join(data_dir,'test')

    #-- load images
    [images,files] = load_data(suffix,ddir)

    sigma = 3
    #-- go through each image and adjust the sigma until a contiuous front is obtained
    for d in ['test']:#,'train']:
        if filter =='canny':
            #-- make output directory
            out_subdir = 'output_canny%s'%suffix
            if (not os.path.isdir(os.path.join(ddir[d],out_subdir))):
                os.mkdir(os.path.join(ddir[d],out_subdir))
            #-- make fronts and save to file    
            for i in range(len(images[d])):

            
                front = feature.canny(images[d][i], sigma=sigma)

                scipy.misc.imsave(os.path.join(ddir[d],out_subdir,'%s'%files[d][i].replace('_Subset',''))\
                    , front)

                if PLOT:    
                    #-- break down image into individual lines by getting indices that are no more than 1 pixel away
                    #-- from each other. Note 'front' is a boolean array (True / 1 for boundary)
                    #-- note that the indices are not in order of radial distance so first we need to sort the points one 
                    #-- by one based on distance
                    indices = np.squeeze(np.nonzero(front)).transpose() # dimensions = num_pts x 2
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
                        if np.squeeze(dist) <=2.:
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
                    new_im = np.zeros(front.shape)
                    for pix_count in range(lens[max_ind]):
                        new_im[np.squeeze(np.squeeze(seg[max_ind])[pix_count])[0],\
                            np.squeeze(np.squeeze(seg[max_ind])[pix_count])[1]] = 1.

                    
                    # display results
                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                                        sharex=True, sharey=True)

                    ax1.imshow(images[d][i], cmap=plt.cm.gray)
                    ax1.axis('off')
                    ax1.set_title('noisy image', fontsize=20)

                    ax2.imshow(front, cmap=plt.cm.gray)
                    ax2.axis('off')
                    ax2.set_title('Canny filter, $\sigma=%i$'%sigma, fontsize=20)

                    ax3.imshow(new_im, cmap=plt.cm.gray)
                    ax3.axis('off')
                    ax3.set_title('longest line', fontsize=20)

                    fig.tight_layout()

                    plt.show()

        elif filter == 'sobel':
            threshold_str = ''
            if threshold != 0:
                threshold_str = '_%.2fthreshold'%threshold
             #-- make output directory
            out_subdir = 'output_sobel%s%s'%(threshold_str,suffix)
            if (not os.path.isdir(os.path.join(ddir[d],out_subdir))):
                os.mkdir(os.path.join(ddir[d],out_subdir))
            #-- make fronts and save to file    
            for i in range(len(images[d])):
                #-- using scikit image sobel filter
                front = sobel(images[d][i])
                #-- invert image colors
                front = 1 - front
                if threshold != 0:
                    #-- set threshold
                    ind = np.where(front >= threshold)
                    front[ind] = 1.
             

                #-- using scipy sobel filter
                #dx=  ndimage.sobel(images[d][i], 0)  # horizontal derivative
                #dy = ndimage.sobel(images[d][i], 1)  # vertical derivative
                #front = np.hypot(dx, dy) 

                #-- using FIL edge detector
                #im = Image.fromarray(np.uint8(images[d][i]*255.))
                #front = im.filter(ImageFilter.FIND_EDGES)

                scipy.misc.imsave(os.path.join(ddir[d],out_subdir,'%s'%files[d][i].replace('_Subset',''))\
                    , front)




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
            run_filter(parameters)

if __name__ == '__main__':
    main()
