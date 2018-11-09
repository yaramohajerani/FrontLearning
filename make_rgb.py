#!/anaconda2/bin/python2.7
u"""
make_rgb.py
by Yara Mohajerani (Last update 11/2018)

Convert images to RGB for manual annotation

Update History
        11/2018 Written
"""
import os
import numpy as np
import imp
import sys
from glob import glob
from PIL import Image
import getopt

def main():
    #-- Read the system arguments listed after the program
    long_options = ['indir=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'=I:',long_options)

    indir = ''
    for opt, arg in optlist:
        if opt in ('-I','--indir'):
            indir = os.path.expanduser(arg)
    

    #-- get a list of the input files
    in_list = sorted([fn for fn in glob(os.path.join(indir,'*png'))
        if (not os.path.basename(fn).endswith('postprocess.png') and not os.path.basename(fn).endswith('threshold.png'))])
    n_files = len(in_list)
    filenames = [os.path.basename(i) for i in in_list]
    print(filenames)
    h,w = np.array(Image.open(in_list[0]).convert('L')).shape

    outdir = os.path.join(os.path.dirname(indir),'output_handrawn')
    #-- make output folder
    if (not os.path.isdir(outdir)):
        os.mkdir(outdir)
    else:
        sys.exit('Files alreay exist. Dont overwrite.')
        
    #-- read files to fix dimensions are remove points dimmer than the threshold
    for i in range(n_files):
        img = Image.open(in_list[i])
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)

        #-- save final image to file
        outfile = os.path.join(outdir,'%s.png'%((filenames[i].replace('_Subset',''))[:-4]))
        rgbimg.save(outfile, 'PNG')

if __name__ == '__main__':
    main()