u"""
extrct_handrawn.py
by Yara Mohajerani (11/2018)

Extract handrawn fronts on the same resolution as the NN was 
trained on for comparison.

History
    11/2018 Written
"""

import os
import numpy as np
import imp
import sys
import getopt
from glob import glob
from PIL import Image
import scipy.misc

#############################################################################################
# All of the functions are run here
#-- main function to get user input and make training data
def main():
    #-- Read the system arguments listed after the program
    long_options = ['indir=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'=I:',long_options)

    indir = ''
    for opt, arg in optlist:
        if opt in ('-I','--indir'):
            indir = os.path.expanduser(arg)
    
    #-- this is only for consistency so it works with other scripts
    threshold_str = 'nothreshold'

    #-- get a list of the input files
    in_list = sorted([fn for fn in glob(os.path.join(indir,'*png'))\
		if (not os.path.basename(fn).endswith('postprocess.png') and not os.path.basename(fn).endswith('threshold.png'))])
    n_files = len(in_list)
    filenames = [os.path.basename(i) for i in in_list]
    print(filenames)
    h,w = np.array(Image.open(in_list[0]).convert('L')).shape

    #-- read files to fix dimensions are remove points dimmer than the threshold
    for i in range(n_files):
        img = np.array(Image.open(in_list[i]))
        reds = img[:,:,0]
        greens = img[:,:,1]
        blues = img[:,:,2]

        #-- extract red points
        ind = np.where((reds==255) & (greens==0) & (blues==0))
        new_im = np.ones((img.shape[0],img.shape[1]))
        new_im[ind] = 0

        #-- save final image to file
        outfile = os.path.join(indir,'%s_%s.png'%((filenames[i].replace('_Subset',''))[:-4],threshold_str))
        scipy.misc.imsave(outfile, new_im)

if __name__ == '__main__':
    main()