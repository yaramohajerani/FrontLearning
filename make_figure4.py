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

prefix = 'LT05_L1TP_233013_19890629_20170202_01_T1_B2' #'LE07_L1TP_233013_20000331_20170212_01_T2_B8'

#fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(6.5, 7.5),sharex=True, sharey=True)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6.5, 4.5),sharex=True, sharey=True)

#-- make custom colormap for final panel (comparison)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] # red, green, blue, white
my_cm = LinearSegmentedColormap.from_list('myColors', colors, N=4)


in_img_file = os.path.join(ddir,'images_equalize_autocontrast_smooth_edgeEnhance',\
    '%s_Subset.png'%prefix)
in_img = np.array(Image.open(in_img_file).convert('L'))/255.


front_file = os.path.join(ddir,'labels','%s_Front.png'%prefix)
front = np.array(Image.open(front_file).convert('L'))/255.



f = os.path.join(ddir,'output_10batches_100epochs_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_noAug = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_60epochs_4layers_32init_82.22weight_w0.2drop_augment-x2_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_augx2 = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_60epochs_4layers_32init_82.22weight_w0.2drop_augment-x3_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_augx3 = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_60epochs_5layers_32init_86.60weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l5_noAug = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_3batches_60epochs_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_3b = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_30batches_100epochs_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_30b = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_10batches_4layers_32init_1.00weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_noweight = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_10batches_60epochs_4layers_64init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_64 = np.array(Image.open(f).convert('L'))/255.

sobel_out_file = os.path.join(ddir,'output_sobel_equalize_autocontrast_smooth_edgeEnhance',\
    '%s.png'%prefix)
sobel_out = np.array(Image.open(sobel_out_file).convert('L'))/255.

cnn_p_file = os.path.join(ddir,'Post Processing Results/CNN HF/CNN HF Post-Processed','%s_Solution.png'%prefix)
cnn_p = np.array(Image.open(cnn_p_file).convert('L'))/255.

sobel_p_file = os.path.join(ddir,'Post Processing Results/Sobel/Sobel Post-Processed','%s_Solution.png'%prefix)
sobel_p = np.array(Image.open(sobel_p_file).convert('L'))/255.

#-- input image
#ax[0,0].imshow(in_img, cmap=plt.cm.gray)
#ax[0,0].set_title(r"$\bf{a)}$" + " Pre-processsed Input", fontsize=10, color='navy')

#-- true front
#ax[0,1].imshow(front, cmap=plt.cm.gray)
#ax[0,1].set_title(r"$\bf{b)}$" + " True Front", fontsize=10, color='navy')

#-- 4 layer standard
ax[0,0].imshow(l4_noAug, cmap=plt.cm.gray)
ax[0,0].set_title(r"$\bf{a)}$" + " No Augmentation", fontsize=10, color='navy')
ax[0,0].text(0.5, 0.1, r"$RMS=%i$ $m$"%np.rint(361.26),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[0,0].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 4 layer aug x2
ax[0,1].imshow(l4_augx2, cmap=plt.cm.gray)
ax[0,1].set_title(r"$\bf{b)}$" + " Augmented:\nMirrored", fontsize=10, color='navy')
ax[0,1].text(0.5, 0.1, r"$RMS=%i$ $m$"%np.rint(225.72),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[0,1].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 4 layer aug x3
ax[0,2].imshow(l4_augx3, cmap=plt.cm.gray)
ax[0,2].set_title(r"$\bf{c)}$" + " Augmented:\nMirrored & Inverted", fontsize=10, color='navy')
ax[0,2].text(0.5, 0.1, r"$RMS=%i$ $m$"%np.rint(283.15),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[0,2].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 5 layer 
ax[1,0].imshow(l5_noAug, cmap=plt.cm.gray)
ax[1,0].set_title(r"$\bf{d)}$" + " 37 Layers\nMax Channels 512\n(No Augmentation)", fontsize=10, color='navy')
ax[1,0].text(0.5, 0.1, r"$RMS=%i$ $m$"%np.rint(296.41),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[1,0].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 4 layer - 64 start 
ax[1,1].imshow(l4_64, cmap=plt.cm.gray)
ax[1,1].set_title(r"$\bf{e)}$" + " 29 Layers\nMax Channels 512\n(No Augmentation)", fontsize=10, color='navy')
ax[1,1].text(0.5, 0.1, r"$RMS=%i$ $m$"%np.rint(358.39),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[1,1].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 4 layer 30 batches
#ax[2,0].imshow(l4_30b, cmap=plt.cm.gray)
#ax[2,0].set_title(r"$\bf{g)}$" + " batch-size 30", fontsize=10, color='navy')

#-- 4 layer 3 batches
ax[1,2].imshow(l4_3b, cmap=plt.cm.gray)
ax[1,2].set_title(r"$\bf{f)}$" + " batch-size 3\n(No Augmentation)", fontsize=10, color='navy')
ax[1,2].text(0.5, 0.1, r"$RMS=%i$ $m$"%np.rint(389.89),
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax[1,2].transAxes,color='black', fontsize=10,style='normal',
        bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 2, 'edgecolor': 'wheat'})

#-- 4 layer no weight
#ax[1,2].imshow(l4_noweight, cmap=plt.cm.gray)
#ax[1,2].set_title(r"$\bf{f)}$" + " No Weights", fontsize=10, color='navy')


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
#fig.tight_layout()
#plt.show()
plt.savefig(os.path.join(main_dir,'Figure_4.pdf'),format='pdf',dpi=300)