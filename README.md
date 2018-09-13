# FrontLearning
This repository contains Python scripts utilizing Keras to identify glacier calving fronts from satellite imagery. 

Scripts by Yara Mohajerani.

Training data provided by Michael Wood.

Two approaches are taken: A Convolution Neural Netowrk (CNN) with a U-Net architecture for image segmentation, and a sliding window
approach for classification of the front. The latter approach is still under development. 

## 1. Image Segmentation with U-Net
![Unet_Logo](./UnetLogo.jpg)

The processing is divided in 3 sections: 

1. pre-processing: `frontlearn_preprocess.py`
  * Configurations are given as commandline arguments.
  * Put data for each glacier in a different folder. Glacier name must be provided in commandline.

2. training and prediction: `unet_train.py` (with option for data augmentation in training)
  * Uses `unet_model.py` to train a U-Net neural network whose architecture is dynamic depending on the specified parameters.
  * Run configurations should be specified in a `.txt` parameter file in command line when executing the script.
3. Post-processing - clean-up and vectorization: `unet_postprocess.py`
  * Removes any noise in the generated calving fronts and vectorizes them as Shapely LineStrings.

Old scripts during the development of this project are kept in `legacy_scripts_obsolete.dir`.

The problem with this approach is that the huge class imbalance between no-boundary and boundary pixels causes the output to be completely blank
(all "no-boundary") when training on a diverse dataset of multiple glaciers. A pixel-to-pixel mapping may not be a the best solution. Thus we also
attempt a sliding window classification. 

## 2. CNN Classification with Sliding Windows

Still under development.

## Some Notes
The advantage of U-Net is that the results are much more accurate and less noisy. However, it's very prone to bias from class imbalance and it is
consiserably slower. One solution could be training the network on sub-patches of the images (similar to sliding windows) where we force half the inputs
the include the boundary and the other half to not include the boundary.
The sliding window approach scales much better with larger data sets and due to weight adjustment does not suffer from class 
imbalance. But the results are yet close enough to the actual target.

More details will be provided as the project moves forward.

For questions contact <ymohajer@uci.edu>.
