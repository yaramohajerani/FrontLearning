u"""
make_results_figure

Plot Figure 3 of paper. Temporary script for figure.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from webcolors import rgb_to_name

#-- directory setup
#- current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.join(current_dir,'..','FrontLearning_data')
glacier_ddir = os.path.join(main_dir,'all_data2.dir')
ddir = os.path.join(glacier_ddir, 'data/test')

prefix = 'LT05_L1TP_233013_19890629_20170202_01_T1_B2' #'LE07_L1TP_233013_20000331_20170212_01_T2_B8'
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(6, 7.),
                                    sharex=True, sharey=True)

#-- make custom colormap for final panel (comparison)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] # red, green, blue, white
my_cm = LinearSegmentedColormap.from_list('myColors', colors, N=4)


in_img_file = os.path.join(ddir,'images_equalize_autocontrast_smooth_edgeEnhance',\
    '%s_Subset.png'%prefix)
in_img = np.array(Image.open(in_img_file).convert('L'))/255.


front_file = os.path.join(ddir,'labels','%s_Front.png'%prefix)
front = np.array(Image.open(front_file).convert('L'))/255.



f = os.path.join(ddir,'output_3batches_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_noAug = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_3batches_4layers_32init_82.22weight_w0.2drop_augment-x2_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_augx2 = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_3batches_4layers_32init_82.22weight_w0.2drop_augment-x3_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_augx3 = np.array(Image.open(f).convert('L'))/255.


f = os.path.join(ddir,'output_10batches_4layers_32init_82.22weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l4_10b = np.array(Image.open(f).convert('L'))/255.

f = os.path.join(ddir,'output_3batches_5layers_32init_86.60weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
    '%s_nothreshold.png'%prefix)
l5 = np.array(Image.open(f).convert('L'))/255.

# f = os.path.join(ddir,'output_10batches_4layers_32init_1.00weight_w0.2drop_equalize_autocontrast_smooth_edgeEnhance_cropped',\
#     '%s_nothreshold.png'%prefix)
# l4_noweight = np.array(Image.open(f).convert('L'))/255.

sobel_out_file = os.path.join(ddir,'output_sobel_equalize_autocontrast_smooth_edgeEnhance',\
    '%s.png'%prefix)
sobel_out = np.array(Image.open(sobel_out_file).convert('L'))/255.

folder = 'output_3batches_4layers_32init_82.22weight_w0.2drop_augment-x3_equalize_autocontrast_smooth_edgeEnhance_cropped'
cnn_p_file = os.path.join(main_dir,'Results/Helheim Results',folder,'%s Post-Processed'%folder,'%s_Solution.png'%prefix)
cnn_p = np.array(Image.open(cnn_p_file).convert('L'))/255.

sobel_p_file = os.path.join(ddir,'Post Processing Results/Sobel/Sobel Post-Processed','%s_Solution.png'%prefix)
sobel_p = np.array(Image.open(sobel_p_file).convert('L'))/255.

#-- input image
ax[0,0].imshow(in_img, cmap=plt.cm.gray)
ax[0,0].set_title(r"$\bf{a)}$" + " Pre-processsed Input", fontsize=12, color='navy')

#-- true front
ax[0,1].imshow(front, cmap=plt.cm.gray)
ax[0,1].set_title(r"$\bf{b)}$" + " True Front", fontsize=12, color='navy')

#-- combine cnn output and true front
#-- combine the 3 output images into one image
out_image = np.ones(front.shape)
ind1 = np.where(cnn_p==0.)
out_image[ind1] = 0.0
ind2 = np.where(sobel_p==0.)
out_image[ind2] = 0.3
ind3 = np.where(front==0.)
out_image[ind3] = 0.6
ax[0,2].imshow(out_image, cmap=my_cm)
ax[0,2].set_title(r"$\bf{c)}$" + " Comparison", fontsize=12, color='navy')
#-- make fake legend
ax[0,2].plot([],[],color=rgb_to_name((255, 0, 0)),label='NN\nPost-processed')
ax[0,2].plot([],[],color=rgb_to_name((0, 255, 0)),label='Sobel\nPost-processed')
ax[0,2].plot([],[],color=rgb_to_name((0, 0, 255)),label='True Front')
ax[0,2].legend(loc='lower center',bbox_to_anchor=(0.5,-0.1),framealpha=1.0)

#-- 4 layer standard
ax[1,0].imshow(l4_noAug, cmap=plt.cm.gray)
ax[1,0].set_title(r"$\bf{d)}$" + " No Augmentation", fontsize=12, color='navy')

#-- 4 layer aug x2
ax[1,1].imshow(l4_augx2, cmap=plt.cm.gray)
ax[1,1].set_title(r"$\bf{e)}$" + " Augmented:\nMirrored", fontsize=12, color='navy')

#-- 4 layer aug x3
ax[1,2].imshow(l4_augx3, cmap=plt.cm.gray)
ax[1,2].set_title(r"$\bf{f)}$" + " Augmented:\nMirrored & Inverted", fontsize=12, color='navy')

#-- 4 layer 10 batches
ax[2,0].imshow(l4_10b, cmap=plt.cm.gray)
ax[2,0].set_title(r"$\bf{f)}$" + " batch-size 10", fontsize=12, color='navy')

#-- 4 layer no weight
#ax[2,1].imshow(l4_noweight, cmap=plt.cm.gray)
#ax[2,1].set_title(r"$\bf{g)}$" + " No Weights", fontsize=12, color='navy')

#-- 5 layer 
ax[2,1].imshow(l5, cmap=plt.cm.gray)
ax[2,1].set_title(r"$\bf{g)}$" + " 37 Layers", fontsize=12, color='navy')

#-- sobel
ax[2,2].imshow(sobel_out, cmap=plt.cm.gray)
ax[2,2].set_title(r"$\bf{h)}$" + " Sobel", fontsize=12, color='navy')

for i in range(3):
    for j in range(3):
        #ax[i,j].axis('off')
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

fig.tight_layout()
plt.savefig(os.path.join(ddir,'Figure_3_v2.pdf'),format='pdf',dpi=300)