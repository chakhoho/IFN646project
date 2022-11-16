# Breast Cancer Mammogrammphy Using VGG CNN Model and Techniques

Rerpot Dissertation for the Master of Information Technology (IFN646 Biomedical Data Science) at the Queensland University of Technology (2022).

The final report can be read and download here: [Breast Cancer Mammogrammphy Classification, Ayush Raj, Chak Ho Chan, Li Jen Shao (2022)](https://github.com/chakhoho/IFN646project/blob/main/group%2010%20-%20IFN646%20report%20finalised%20.pdf)

## Abstract

Deep learning-based neural network advances made recently in biological image processing might be used to increase the efficiency of Computer Aided Diagnosis (CAD) systems. An overview of the most current cutting-edge deep learning-based CAD systems created for mammography and breast histopathology pictures is provided, considering the significance of breast cancer globally and the promising outcomes reported by VGG back propagation-based approach in breast imaging. The study describes how well a mammographic imaging can forecast a positive patient for breast cancer while considering the breast tissue textural characteristics such as energy, contrast, correlation, and other texture descriptors at each pixel. The VGG Neural Network Model, which we suggest as a computer-based method to modelling breast cancer, classifies the image as either normal tissue, Benign or Malignant tumour. We reached an accuracy of nearly 65% for the VGG-Neural Network model on the given test set and the model was found to be a good fit for classification. 


## Acknowledgments

We thank Prof. Dimitri Perrin and Dr. Jake Bradford for giving us this project and providing us the guidance and support for all our work. 

## Problem Statement

Breast cancer continues to be the second largest cause of cancer death worldwide and is the most frequent cancer in women. The abnormal development of the cells lining the breast lobules or ducts is breast cancer. These cells have the capacity to spread to many bodily areas and multiply uncontrolled. The most common symptom found was breast thickening or new lumps, especially if they are present in only one breast.

Most of the major causes of breast cancer are genetic factors - damaged DNA and family history. However, other risk factors may be related to lifestyle or environment, such as alcohol consumption - studies have shown that women who drink three drinks a day are 1.5 times more likely to be affected, obesity, hormone therapy - increased estrogen levels due to treatment with hormone replacement pills may be associated with breast cancer, and sedentary inactivity. Other causes, such as having children later in life or improper breastfeeding, may also be responsible.
However, the biggest cause is lack of awareness, treatment and screening methods. The lack of specialized radiologists and diagnostic centers and the delay in providing the necessary care is a major problem. To help this growing cause, we aim to develop deep learning models to detect suspicious lesions and thus provide timely and effective diagnosis.

## Datasets

### CBIS-DDSM
The DDSM (Digital Database for Screening Mammography) dataset is one of the most famous databases for breast mammographic research. It is a resource popularly used by the entire mammographic image analysis research community. Primary support for this project was a grant from the Breast Cancer Research Program of the U.S. Army Medical Research and Materiel Command. The Massachusetts General Hospital, the University of South Florida, and Sandia National Laboratories have also contributed. Additional cases were provided from Washington University School of Medicine. The dataset contains nearly 2500 studies with 12 volumes of normal images, containing 695 cases; 15 volumes of cancerous, containing 855 cases; 14 volumes benign, containing 870 cases; and 2 volumes of benign without callback, containing 141 cases.


## Proposed VGG style conventional neural network

VGG Neural Networks. In most cases, it alludes to a deep convolutional network for object identification that performed exceptionally well on the ImageNet dataset. Currently it is the most capable model for object detection. Key features include using ReLU activation function in-place of tanh function, optimization for multiple GPU’s and overlapping pooling. Also, it does address overfitting by using data augmentation. It also improved the traditional CNN model on training image data.


### GRADCAM

In general, we start with a picture as our input and build a model that is stopped at the layer for which we wish to build a Grad-CAM heat-map. For prediction, we affix the completely linked layers. The model is then applied to the input before the layer output and loss are collected. The gradient of the output of our chosen model layer with respect to the model loss is then determined. To overlay the heat-map with the original picture, we next take portions of the gradient that contribute to the prediction and decrease, resize, and rescale them.


## Gradient based saliency maps

They are a well-liked visualisation technique for understanding why a deep learning network chose a particular action, like categorising an image. The gradient expresses how much a variable may influence the outcome of another variable.
Saliency maps are typically shown as heatmaps, with hotness corresponding to regions with a significant influence on the model's choice.

## Method 

The mammography are obtained from CBIS-DDSM(Curated Breast Imaging Subset of DDSM), it is an an updated and standardized version of the Digital Database for Screening Mammography(DDSM), it contains normal, benign, and malignant cases with verified pathology information. Initially, the DDSM is a database of 2,620 scanned film mammography studies with 9684 images. Initially,  the data size is over 160GB, due to the hardware limitation we used the processed version of CBIS-DDSM from [Kaggle](https://www.kaggle.com/datasets/cheddad/miniddsm). This MINI-DDSM dataset contains all the mammograms from the original CBIS-DDSM database, and it has been converted from ddsm file to png file. There are 3 kinds of images which are normal mammograms, mammograms with benign tissue and mammograms with malignant tissue, here we want to focus on right breast CC view images only due to hardware limitation. The goal of this analysis is to train a deep learning model and classify the  mammography into into 3 classes (Normal, Benign and Malignant). The datasource can be found in the releases [project](https://github.com/chakhoho/IFN646project/releases/tag/hostedfile1.0).


## Dependencies

- pandas==1.4.2
- numpy==1.22.4
- cv2==4.6.0
- PIL==9.0.1
- seaborn==0.11.2
- tensorflow==2.10.0
- sklearn==1.1.3
- tensorflow.keras==2.10.0


