#!/anaconda2/bin/python2.7
u"""
crop_output.py
by Yara Mohajerani

remove padding and recrop to get back to original size. 
remove faint points

History
    09/2018 Written
"""
import os
import numpy as np
import imp
import sys
from glob import glob
from PIL import Image
import scipy.misc

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

    print subdir

    #-- also get the unpadded but cropped directory
    unpadded_subdir = os.path.join(tst_dir,'images%s%s'%(suffix,crop_str))
    #-- get a sample file for dimensions
    uncropped_list = glob(os.path.join(unpadded_subdir,'*.png'))
    orig_shape = np.array(Image.open(uncropped_list[0]).convert('L')).shape
    

    #-- get a list of the input files
    in_list = sorted([fn for fn in glob(os.path.join(subdir,'*png'))
         if (not os.path.basename(fn).endswith('postprocess.png') and not os.path.basename(fn).endswith('threshold.png'))])
    n_files = len(in_list)
    filenames = [os.path.basename(i) for i in in_list]
    print(filenames)
    h,w = np.array(Image.open(in_list[0]).convert('L')).shape

    #-- cropping, same numbers as hard coded in crop_input 
    #-- NOTE BE CAREFUL WITH THIS LATER. SHOULD MAKE INTO PARAMETER!
    hcrop,wcrop = 30,25

    #-- read files to fix dimensions are remove points dimmer than the threshold
    for i in range(n_files):
        img = np.array(Image.open(in_list[i]).convert('L'))/255.

        if at != 0.:
            #-- clean up points below the threshold
            img_flat = img.flatten()
            ind_black = np.squeeze(np.nonzero(img_flat <= at))
            ind_white = np.squeeze(np.nonzero(img_flat > at))
            img_flat[ind_black] = 0.
            img_flat[ind_white] = 1.
            img = img_flat.reshape(img.shape)

        #-- remove extra cropping that was done for pooling
        img_nopad = np.ones(orig_shape)
        img_nopad =  img[:orig_shape[0],:orig_shape[1]]

        #-- finally redo the cropping back to the original
        h_final = orig_shape[0]+2*hcrop
        w_final = orig_shape[1]+2*wcrop
        img_final = np.ones((h_final,w_final))

        img_final[hcrop:h_final-hcrop,wcrop:w_final-wcrop] = img_nopad[:,:]

        #-- save final image to file
        outfile = os.path.join(subdir,'%s_%s.png'%(filenames[i][:-4],threshold_str))
        scipy.misc.imsave(outfile, img_final)

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
