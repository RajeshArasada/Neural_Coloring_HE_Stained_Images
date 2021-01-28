# Dillinger

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
# Neural Staining of H&E images 

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
## Problem Definition
I developed a method to automatically transfer the staining information from the training data to new test images in colorectal cancer tissue sections based on convolutional neural networks (CNNs). Application of CNNs to hematoxylin and eosin (H&E) stained histological tissue sections is hampered by: (1) noisy and expensive reference standards established by pathologists and (2) lack of generalization due to staining variation across laboratories,

### **This is an ongoing projects and the results presented here are not final.

```
Domain      : Machine Learning
Techniques  : Convolution Neural Network
Application : Healthcare Management
```

## Dataset Details
Dataset Links: [NCT-CRC-HE-100K](https://zenodo.org/record/1214456), [CRC-VAL-HE-7K](https://zenodo.org/record/1214456)

Original Dataset		: Kather JN, Krisam J, Charoentong P, Luedde T, Herpel E, Weis C-A, et al. (2019) Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. PLoS Med 16(1): e1002730. 

![Model_Architecture](/figures/neural_staining_model_architecture.png)

## Prediction
### Sample Predictions - 01
![Sample_prediction_1](/figures/sample_prediction_01.png)
### Sample Predictions - 02
![Sample_prediction_1](/figures/sample_prediction_01.png)
### Sample Predictions - 03
![Sample_prediction_1](/figures/sample_prediction_01.png)

## Model Evaluation
# Classification of true images
![Confusion_Matrix_True_Images](/figures/confusion_matrix_true_images.png)

# Classification of retsained images
![Confusion_Matrix_True_Images](/figures/confusion_matrix_colored_images.png)


## Tools/ Libraries
```
Languages	: Python
Tools/IDE	: Jupyter Notebook
Libraries	: Tensorflow2.0, scikit learn, Pandas, Numpy, Matplotlib, VGG19
```
