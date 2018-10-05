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

batch = 3
#-- directory setup
#- current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.join(current_dir,'..','FrontLearning_data')
glacier_ddir = os.path.join(main_dir,'all_data2.dir')
ddir = os.path.join(glacier_ddir, 'data/test')

prefix_list = ['LT05_L1TP_233013_19890629_20170202_01_T1_B2',\
    'LE07_L1GT_233013_20010318_20170206_01_T2_B8',\
    'LC08_L1TP_233013_20150707_20170407_01_T1_B8']
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(9, 6.5),
                                    sharex=True, sharey=True)

#-- make custom colormap for final panel (comparison)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] # red, green, blue, white
my_cm = LinearSegmentedColormap.from_list('myColors', colors, N=4)

for i,prefix,j in zip([0,1,2],prefix_list,[5,7,8]):
    in_img_file = os.path.join(ddir,'images_equalize_autocontrast_smooth_edgeEnhance',\
        '%s_Subset.png'%prefix)
    in_img = np.array(Image.open(in_img_file).convert('L'))/255.

    if batch==10:
        cnn_out_file = os.path.join(ddir,'output_10batches_60epochs_4layers_32init_82.22weight_w0.2drop_augment-x2_equalize_autocontrast_smooth_edgeEnhance_cropped',\
            '%s_nothreshold.png'%prefix)
    else:
        cnn_out_file = os.path.join(ddir,'output_3batches_4layers_32init_82.22weight_w0.2drop_augment-x2_equalize_autocontrast_smooth_edgeEnhance_cropped',\
            '%s_nothreshold.png'%prefix)
    cnn_out = np.array(Image.open(cnn_out_file).convert('L'))/255.

    sobel_out_file = os.path.join(ddir,'output_sobel_equalize_autocontrast_smooth_edgeEnhance',\
        '%s.png'%prefix)
    sobel_out = np.array(Image.open(sobel_out_file).convert('L'))/255.

    cnn_p_file = os.path.join(ddir,'Post Processing Results/CNN HF/CNN HF Post-Processed','%s_Solution.png'%prefix)
    cnn_p = np.array(Image.open(cnn_p_file).convert('L'))/255.

    sobel_p_file = os.path.join(ddir,'Post Processing Results/Sobel/Sobel Post-Processed','%s_Solution.png'%prefix)
    sobel_p = np.array(Image.open(sobel_p_file).convert('L'))/255.

    front_file = os.path.join(ddir,'labels','%s_Front.png'%prefix)
    front = np.array(Image.open(front_file).convert('L'))/255.

    #-- combine the 3 output images into one image
    out_image = np.ones(front.shape)
    ind1 = np.where(cnn_p==0.)
    out_image[ind1] = 0.0
    ind2 = np.where(sobel_p==0.)
    out_image[ind2] = 0.3
    ind3 = np.where(front==0.)
    out_image[ind3] = 0.6

    ax[i,0].imshow(in_img, cmap=plt.cm.gray)
    ax[i,0].axes.get_xaxis().set_ticks([])
    ax[i,0].axes.get_yaxis().set_ticks([])
    ax[i,0].set_ylabel('Landsat %i'%j, fontsize=15)

    ax[i,1].imshow(cnn_out, cmap=plt.cm.gray)
    ax[i,1].axes.get_xaxis().set_ticks([])
    ax[i,1].axes.get_yaxis().set_ticks([])

    ax[i,2].imshow(sobel_out, cmap=plt.cm.gray)
    ax[i,2].axes.get_xaxis().set_ticks([])
    ax[i,2].axes.get_yaxis().set_ticks([])

    ax[i,3].imshow(out_image, cmap=my_cm)
    ax[i,3].axes.get_xaxis().set_ticks([])
    ax[i,3].axes.get_yaxis().set_ticks([])
   
    if i==0:
        ax[i,0].set_title('Pre-processsed Input', fontsize=15)
        ax[i,1].set_title('NN Output', fontsize=15)
        ax[i,2].set_title('Sobel Output', fontsize=15)
        ax[i,3].set_title('Processed Comparison', fontsize=15)

         #-- make fake legend
        ax[i,3].plot([],[],color=rgb_to_name((255, 0, 0)),label='NN')
        ax[i,3].plot([],[],color=rgb_to_name((0, 255, 0)),label='Sobel')
        ax[i,3].plot([],[],color=rgb_to_name((0, 0, 255)),label='True Front')
        ax[i,3].legend(loc='upper right', bbox_to_anchor=(0.95, 0.5))

fig.tight_layout()
plt.savefig(os.path.join(ddir,'Figure_3_v3_batch%i.pdf'%batch),format='pdf',dpi=300)