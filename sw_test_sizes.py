#!/anaconda2/bin/python2.7
u"""
test_sizes.py
Temporary script to experiment with different window sizes and proportions of labels
"""

import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

#-- set to True if want to look at plots
PLOT = False
glacier = 'Sverdrup'

#-- directory setup
current_dir = os.path.dirname(os.path.realpath(__file__))
ddir = os.path.join(current_dir,'%s.dir'%glacier)
data_dir = os.path.join(ddir, 'data')
trn_dir = os.path.join(data_dir,'train')
tst_dir = os.path.join(data_dir,'test')

#-- test sizes
f1 = os.path.join(tst_dir,'images_autocontrast_edgeEnhance/LT05_L1TP_024006_19870920_20170211_01_T1_Front.png')
f2 = os.path.join(tst_dir,'labels/LT05_L1TP_024006_19870920_20170211_01_T1_Front.png')

img = np.array(Image.open(f1).convert('L'))/255.
lbl = np.array(Image.open(f2).convert('L'))/255.
print img.shape
HH,HW = 20,20
tot_pixels = (2*HH+1)*(2*HW+1)

if PLOT:
    n_windows = 10
    #-- take n_window random samples from the image
    ih = np.random.randint(HH,high=img.shape[0]-HH,size=n_windows)
    iw = np.random.randint(HW,high=img.shape[1]-HW,size=n_windows)
    for j in range(n_windows):
        plt.imshow(img[ih[j]-HH:ih[j]+HH+1,iw[j]-HW:iw[j]+HW+1])
        label_box = lbl[ih[j]-HH:ih[j]+HH+1,iw[j]-HW:iw[j]+HW+1]
        count_boundary = np.count_nonzero(label_box==0.)
        #-- if there is any pixe
        # l from a boundary, mark as boundary
        if count_boundary > 0.01*tot_pixels:
            lbl_text = "YES"
        else:
            lbl_text = "FALSE"
        plt.title(lbl_text)
        plt.show()

#-- repeat but this time get percentage of posittives
n_windows = 300
count_true = 0
count_false = 0
#-- take n_window random samples from the image
ih = np.random.randint(HH,high=img.shape[0]-HH,size=n_windows)
iw = np.random.randint(HW,high=img.shape[1]-HW,size=n_windows)
for j in range(n_windows):
    plt.imshow(img[ih[j]-HH:ih[j]+HH+1,iw[j]-HW:iw[j]+HW+1])
    label_box = lbl[ih[j]-HH:ih[j]+HH+1,iw[j]-HW:iw[j]+HW+1]
    count_boundary = np.count_nonzero(label_box==0.)
    #-- if there is any pixe
    # l from a boundary, mark as boundary
    if count_boundary > 0.01*tot_pixels:
        count_true += 1
    else:
        count_false += 1

print 'Boundary percentage = %.2f'%(100.*np.float(count_true)/n_windows)
print 'No boundary percentage = %.2f'%(100.*np.float(count_false)/n_windows)
print 'no boundary : boudary ratio = %.2f'%(np.float(count_false)/np.float(count_true))