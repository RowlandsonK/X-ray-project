# Chest X-ray Image Classification with Deep Learning

## Overview
This project focuses on the classification of Chest X-ray images into two categories: "Normal" and "Pneumonia" using deep learning models. The primary objective is to leverage convolutional neural networks (CNNs) to develop robust classifiers capable of identifying signs of pneumonia in medical images.


### Table of Contents
1.	Introduction
2.	Dataset
3.	Exploratory Data Analysis (EDA) and Data Preparation
4.	Data Augmentation
5.	Modelling
6.	Model Evaluation
7.	Results and Visualizations
8.	Usage
9.	Dependencies
10.	Future Improvements
11.	Acknowledgements
#### Introduction

Chest X-ray examinations are vital for diagnosing various respiratory conditions, with pneumonia being a significant health concern. This project aims to automate the identification of pneumonia from X-ray images, contributing to timely and accurate medical diagnoses.
Dataset

The dataset comprises Chest X-ray images categorized into "Normal" and "Pneumonia." The 41st and 88th images from the training set were selected for exploratory analysis, revealing variations in image sizes (1604x1248 and 1072x712, respectively). The need for standardization prompted image resizing to a target size of (1000x1000). The dataset was then split into training, validation, and test sets.

##### Exploratory Data Analysis (EDA) and Data Preparation

Exploratory data analysis confirmed the initial size disparity and motivated the resizing process. Images were loaded, converted to grayscale, and visualized for both "Normal" and "Pneumonia" classes. The standardized images were saved in new directories for further processing.
Data Augmentation

To enhance model generalization, data augmentation techniques were applied, including rotation, width/height shift, shear, zoom, and horizontal flip. The ImageDataGenerator from Keras facilitated these transformations.

###### Modelling
The VGG16 architecture was selected as the baseline model. The pre-trained VGG16 model, trained on ImageNet, was utilized for feature extraction. The top layers were modified to include a Flatten layer, Dense layers with ReLU activation, Dropout, and a Softmax output layer. The model was compiled using the Adam optimizer and categorical crossentropy loss.

####### Model Evaluation
The VGG16 model achieved a test accuracy of 88.16%, indicating strong predictive capabilities. The confusion matrix highlighted the model's proficiency in pneumonia detection but revealed room for improvement, particularly in reducing false positives and false negatives.

######## Results and Visualizations
Visualizations included training/validation accuracy and loss plots, along with a confusion matrix heatmap. The close alignment of training and validation metrics demonstrated the model's generalization to unseen data.
Usage

To reproduce the experiments, follow the provided Jupyter Notebook link, ensuring the required dependencies are installed. Adjustments to parameters and architecture can be made for further experimentation.

######### Dependencies
•	Python 3.x
•	TensorFlow
•	Keras
•	Matplotlib
•	NumPy
•	Pandas
•	Seaborn

########## Future Improvements

Potential areas for improvement include fine-tuning, additional data collection, and exploring alternative architectures to enhance the model's performance.
Acknowledgements

The project utilizes a Chest X-ray dataset, and the code is inspired by best practices in deep learning and transfer learning. Special thanks to the authors and contributors of the used datasets and libraries.

