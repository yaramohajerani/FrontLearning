u"""
make_results_figure

Plot Figure 3 of paper. Temporary script for figure.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#-- directory setup
#- current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.join(current_dir,'..','FrontLearning_data')
glacier_ddir = os.path.join(main_dir,'all_data2.dir')
ddir = os.path.join(glacier_ddir, 'data/test')

prefix = 'LE07_L1TP_233013_20000331_20170212_01_T2_B8'
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(9, 6),
                                    sharex=True, sharey=True)

#-- make custom colormap for final panel (comparison)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] # red, green, blue, white
my_cm = LinearSegmentedColormap.from_list('myColors', colors, N=4)


in_img_file = os.path.join(ddir,'images_equalize_autocontrast_smooth_edgeEnhance',\
    '%s_Subset.png'%prefix)
in_img = np.array(Image.open(in_img_file).convert('L'))/255.


front_file = os.path.join(ddir,'labels','%s_Front.png'%prefix)
front = np.array(Image.open(front_file).convert('L'))/255.



f = os.path.join(ddir,'output_10batches_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_noAug = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_4layers_32init_82.22weight_w0.2drop_augment-x2_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_augx2 = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_4layers_32init_82.22weight_w0.2drop_augment_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_augx3 = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_5layers_40init_86.60weight_w0.5drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l5_noAug = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_3batches_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_3b = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_30batches_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_30b = np.array(Image.open(f).convert('L'))/255.


sobel_out_file = os.path.join(ddir,'output_sobel_equalize_autocontrast_smooth_edgeEnhance',\
    '%s.png'%prefix)
sobel_out = np.array(Image.open(sobel_out_file).convert('L'))/255.

#-- input image
ax[0,1].imshow(in_img, cmap=plt.cm.gray)
ax[0,1].axis('off')
ax[0.1].set_title('Pre-processsed Input', fontsize=15)

#-- true front
ax[0,2].imshow(front, cmap=plt.cm.gray)
ax[0,2].axis('off')
ax[0,2].set_title('True Front', fontsize=15)

#-- 4 layer standard
ax[1,0].imshow(l4_noAug, cmap=plt.cm.gray)
ax[1,0].axis('off')
ax[1,0].set_title('29 Layers - No Augmentation -  10 batchs', fontsize=15)

#-- 4 layer aug x2
ax[1,1].imshow(l4_augx2, cmap=plt.cm.gray)
ax[1,1].axis('off')
ax[1,1].set_title('29 Layers - Augmentation: Horizontal Flip', fontsize=15)

#-- 4 layer aug x3
ax[1,2].imshow(l4_augx3, cmap=plt.cm.gray)
ax[1,2].axis('off')
ax[1,2].set_title('29 Layers - Augmentation: Horizontal Flip & Inversion', fontsize=15)

#-- 4 layer standard
ax[1,3].imshow(l5_noAug, cmap=plt.cm.gray)
ax[1,3].axis('off')
ax[1,3].set_title('37 Layers - No Augmentation ', fontsize=15)




fig.tight_layout()
plt.show()
#plt.savefig(os.path.join(ddir,'Figure_3b.pdf'),format='pdf',dpi=300)