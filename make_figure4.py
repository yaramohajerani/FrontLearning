#!/usr/bin/env python
u"""
make_figure4.py

Plot Figure 4 of paper.
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
glacier_ddir = os.path.join(main_dir,'greenland_training.dir')
ddir = os.path.join(glacier_ddir, 'data','test')

prefix = 'LT05_L1TP_233013_19890629_20170202_01_T1_B2'

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6.5, 4.5),sharex=True, sharey=True)

#-- make custom colormap for final panel (comparison)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] # red, green, blue, white
my_cm = LinearSegmentedColormap.from_list('myColors', colors, N=4)


in_img_file = os.path.join(ddir,'images_equalize_autocontrast_smooth_edgeEnhance',\
    '%s_Subset.png'%prefix)
in_img = np.array(Image.open(in_img_file).convert('L'))/255.


front_file = os.path.join(ddir,'labels','%s_Front.png'%prefix)
front = np.array(Image.open(front_file).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_100epochs_4layers_32init_241.15weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped_1px',\
    '%s_nothreshold.png'%prefix)
l4_noAug = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_60epochs_4layers_32init_241.15weight_w0.2drop_augment-x2_equalize_autocontrast_smooth_edgeEnhance_cropped_1px',\
    '%s_nothreshold.png'%prefix)
l4_augx2 = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_60epochs_4layers_32init_241.15weight_w0.2drop_augment-x3_equalize_autocontrast_smooth_edgeEnhance_cropped_1px',\
    '%s_nothreshold.png'%prefix)
l4_augx3 = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_100epochs_5layers_32init_253.89weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped_1px',\
    '%s_nothreshold.png'%prefix)
l5_noAug = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_3batches_30epochs_4layers_32init_241.15weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped_1px',\
    '%s_nothreshold.png'%prefix)
l4_3b = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_10batches_60epochs_4layers_64init_241.15weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped_1px',\
    '%s_nothreshold.png'%prefix)
l4_64 = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_10batches_100epochs_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)

l4_3pix = np.array(Image.open(f).convert('L'))/255.

sobel_out_file = os.path.join(ddir,'output_sobel_equalize_autocontrast_smooth_edgeEnhance',\
    '%s.png'%prefix)
sobel_out = np.array(Image.open(sobel_out_file).convert('L'))/255.


#-- 4 layer standard
ax[0,0].imshow(l4_noAug, cmap=plt.cm.gray)
ax[0,0].set_title(r"$\bf{a)}$" + " No Augmentation", fontsize=10, color='navy')
ax[0,0].text(0.5, 0.1, r"$\epsilon=%i$ $m$"%np.rint(106.99),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[0,0].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 4 layer aug x2
ax[0,1].imshow(l4_augx2, cmap=plt.cm.gray)
ax[0,1].set_title(r"$\bf{b)}$" + " Augmented:\nMirrored", fontsize=10, color='navy')
ax[0,1].text(0.5, 0.1, r"$\epsilon=%i$ $m$"%np.rint(96.31),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[0,1].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 4 layer aug x3
ax[0,2].imshow(l4_augx3, cmap=plt.cm.gray)
ax[0,2].set_title(r"$\bf{c)}$" + " Augmented:\nMirrored & Inverted", fontsize=10, color='navy')
ax[0,2].text(0.5, 0.1, r"$\epsilon=%i$ $m$"%np.rint(137.88),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[0,2].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 4 layer - 3 pixel label 
ax[1,0].imshow(l4_3pix, cmap=plt.cm.gray)
ax[1,0].set_title(r"$\bf{d)}$" + " 3-pixel Label\n(No Augmentation)", fontsize=10, color='navy')
ax[1,0].text(0.5, 0.1, r"$\epsilon=%i$ $m$"%np.rint(170.48),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[1,0].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})


#-- 4 layer 3 batches
ax[1,1].imshow(l4_3b, cmap=plt.cm.gray)
ax[1,1].set_title(r"$\bf{e)}$" + " batch-size 3\n(No Augmentation)", fontsize=10, color='navy')
ax[1,1].text(0.5, 0.1, r"$\epsilon=%i$ $m$"%np.rint(152.30),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[1,1].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 5 layer 
ax[1,2].imshow(l5_noAug, cmap=plt.cm.gray)
ax[1,2].set_title(r"$\bf{f)}$" + " 37 Layers\nMax Channels 512\n(No Augmentation)", fontsize=10, color='navy')
ax[1,2].text(0.5, 0.1, r"$\epsilon=%i$ $m$"%np.rint(112.00),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[1,2].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

for i in range(2):
    for j in range(3):
        ax[i,j].axis('equal')
        ax[i,j].get_xaxis().set_ticks([])
        ax[i,j].get_yaxis().set_ticks([])
        ax[i,j].spines['top'].set_visible(True)
        ax[i,j].spines['right'].set_visible(True)
        ax[i,j].spines['bottom'].set_visible(True)
        ax[i,j].spines['left'].set_visible(True)
        ax[i,j].spines['bottom'].set_color('0.5')
        ax[i,j].spines['top'].set_color('0.5')
        ax[i,j].spines['right'].set_color('0.5')
        ax[i,j].spines['left'].set_color('0.5')

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

plt.savefig(os.path.join(main_dir,'Figure_4.pdf'),format='pdf',dpi=300)
