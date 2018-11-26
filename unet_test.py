#!/anaconda2/bin/python2.7
u"""
unet_test.py
by Yara Mohajerani (Last Update 11/2018)

Test U-Net model in frontlearn_unet.py

Update History
    11/2018 Fork from unet_train
"""
import os
import numpy as np
import keras
from keras.preprocessing import image
import imp
import sys
from glob import glob
from PIL import Image,ImageOps
from keras import backend as K
from tensorflow.python.client import device_lib
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight

#-- Print backend information
print(device_lib.list_local_devices())
print(K.tensorflow_backend._get_available_gpus())

#-- read in images
def load_data(suffix,tst_dir,n_layers):
    #-- make subdirectories for input images
    tst_subdir = os.path.join(tst_dir,'images%s'%(suffix))
    #-- get a list of the input files
    tst_list = glob(os.path.join(tst_subdir,'*.png'))
    #-- get just the file names
    tst_files = [os.path.basename(i) for i in tst_list]

    #-- read training data
    n = len(tst_files)
    #-- get dimensions, force to 1 b/w channel
    im_shape = np.array(Image.open(tst_list[0]).convert('L')).shape
    h,w = im_shape
    # pad height and width until it's at least divisible by the right number for the given
    # network depth
    n_div = 2**(n_layers-1)
    if h%n_div != 0:
        h_pad = h+n_div-(h%n_div)
    else:
        h_pad = np.copy(h)
    if w%n_div != 0:
        w_pad = w+n_div-(w%n_div)
    else:
        w_pad = np.copy(w)
  
    #-- get the test data
    n_test = len(tst_files)
    test_img = np.ones((n_test,h_pad,w_pad))
    for i in range(n_test):
        test_img[i][:im_shape[0],:im_shape[1]] = np.array(Image.open(tst_list[i]).convert('L'))/255.

    return {'tst_img':test_img.reshape(n_test,h_pad,w_pad,1),'tst_names':tst_files,'orig_shape':im_shape}

#-- train model and make predictions
def train_model(parameters):
    glacier = parameters['GLACIER_NAME']
    model_glacier =parameters['MODEL_DIR']
    n_batch = int(parameters['BATCHES'])
    n_epochs = int(parameters['EPOCHS'])
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
        aug_config = np.int(parameters['AUG_CONFIG'])
        aug_str = '_augment-x%i'%aug_config
    else:
        aug_config = 0
        aug_str = ''
    
    if parameters['CROP'] in ['Y','y']:
        crop_str = '_cropped'
    else:
        crop_str = ''
    
    if parameters['NORMALIZE'] in ['y','Y']:
        normalize = True
        norm_str = '_normalized'
    else:
        normalize = False
        norm_str = ''
    
    if parameters['LINEAR'] in ['Y','Y']:
        linear = True
        lin_str = '_linear'
    else:
        linear = False
        lin_str = ''

    drop_str = ''
    if drop>0:
        drop_str = '_w%.1fdrop'%drop

    #-- width of labels (pixels)
    #-- don't label 3-pix width to be consistent with old results
    if parameters['LABEL_WIDTH'] == '3':
        lbl_width = ''
    else:
        lbl_width = '_%ipx'%int(parameters['LABEL_WIDTH'])

    if (normalize) and (drop!=0):
        sys.exit('Both batch normalization and dropout are selecte. Choose one.')

    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.path.join(current_dir,'..','FrontLearning_data')
    glacier_ddir = os.path.join(main_dir,'%s.dir'%glacier)
    model_dir = os.path.join(main_dir,'%s.dir'%model_glacier)
    data_dir = os.path.join(glacier_ddir, 'data')
    trn_dir = os.path.join(data_dir,'train')
    tst_dir = os.path.join(data_dir,'test')

    #-- load images
    data = load_data(suffix,tst_dir,n_layers)

    n,height,width,channels=data['tst_img'].shape
    print('width=%i'%width)
    print('height=%i'%height)

    #-- import mod
    unet = imp.load_source('unet_model', os.path.join(current_dir,'unet_model.py'))
    if normalize:
        if linear:
            model = unet.unet_model_linear_normalized(height=height,width=width,channels=channels,\
                n_init=n_init,n_layers=n_layers)
            print('importing unet_model_linear_normalized')
        else:
            model = unet.unet_model_double_normalized(height=height,width=width,channels=channels,\
                n_init=n_init,n_layers=n_layers)
            print('importing unet_model_double_normalized')
    else:
        if linear:
            model = unet.unet_model_linear_dropout(height=height,width=width,channels=channels,\
                n_init=n_init,n_layers=n_layers,drop=drop)
            print('importing unet_model_linear_dropout')
        else:
            model = unet.unet_model_double_dropout(height=height,width=width,channels=channels,\
                n_init=n_init,n_layers=n_layers,drop=drop)
            print('importing unet_model_double_dropout')

    #-- checkpoint file
    chk_file = os.path.join(model_dir,'unet_model_weights_%ibatches_%iepochs_%ilayers_%iinit%s%s%s%s%s%s%s%s.h5'\
        %(n_batch,n_epochs,n_layers,n_init,lin_str,imb_str,drop_str,norm_str,aug_str,suffix,crop_str,lbl_width))
    print('model file:%s'%chk_file)

    #-- if file exists, read model from file
    if os.path.isfile(chk_file):
        print('Check point exists; loading model from file.')
        # load weights
        model.load_weights(chk_file)
    else:
        sys.exit('Model not found.')

    print('Running on test data...')

   
    out_imgs = model.predict(data['tst_img'], batch_size=1, verbose=1)
    print out_imgs.shape
    out_imgs = out_imgs.reshape(out_imgs.shape[0],height,width,out_imgs.shape[2])
    print out_imgs.shape
    #-- make output directory
    out_subdir = 'output_%s_%ibatches_%iepochs_%ilayers_%iinit%s%s%s%s%s%s%s%s'\
        %(model_glacier,n_batch,n_epochs,n_layers,n_init,lin_str,imb_str,drop_str,norm_str,aug_str,suffix,crop_str,lbl_width)
    if (not os.path.isdir(os.path.join(tst_dir,out_subdir))):
        os.mkdir(os.path.join(tst_dir,out_subdir))
    #-- save the test image
    for i in range(len(out_imgs)):
        if at != 0.:
            #-- clean up points below the threshold
            img_flat = out_imgs[i].flatten()
            ind_black = np.squeeze(np.nonzero(img_flat <= at))
            ind_white = np.squeeze(np.nonzero(img_flat > at))
            img_flat[ind_black] = 0.
            img_flat[ind_white] = 1.
            img_array = img_flat.reshape(out_imgs[i].shape)
            #-- convert back to original dimension
            img_final = img_array[:data['orig_shape'][0],:data['orig_shape'][1]]
        else:
            img_final = out_imgs[i][:data['orig_shape'][0],:data['orig_shape'][1]]

        #-- convert to image
        im = image.array_to_img(img_final)
        #im = ImageOps.autocontrast(image.array_to_img(out_imgs[i]))
        out_name = '%s_%s.png'%((data['tst_names'][i].replace('_Subset',''))[:-4],threshold_str)
        #out_name = '%s.png'%((data['tst_names'][i].replace('_Subset',''))[:-4])
        print(os.path.join(tst_dir,out_subdir,out_name)) 
        im.save(os.path.join(tst_dir,out_subdir,out_name))

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
            train_model(parameters)

if __name__ == '__main__':
    main()
