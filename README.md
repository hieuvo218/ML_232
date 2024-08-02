# ML_232
## Overview
This project focuses on the image classification task using MNIST database. 
## Dataset
MNIST dataset consists of 60,000 training images and 10,000 testing images. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.
## Programming Languages and Packages used
- Python
- Scikit Learn
- Keras
- Numpy
- Seaborn
- Matplotlib
## Setup and Installation
Install all the required packages, dataset get from keras.
## Algorithms
- K-nearest neighbors.
- Support vector machine.
- Naive Bayes classifier.
- Gradient Boosting
- Convolutional neural network.
## Note
This repository contains images, a report, many python files and one notebook.
- NaiveBayes.py: This file implements Naive Bayes classifier, a class named 'Dataset' to load data and a function to calculate probability distribution.
- cnn_for_mnist.py: Convolutional Neural Network to classify images
- MNIST_classification_with_several_learning_techniques.ipynb: A notebook to implement machine learning methods. In this notebook, I use t-SNE technique to have a look at the dataset. For classifying, I contrucsted multiple models based on different learning algorithms to test their efficiency. In constructing SVM, I separate train dataset to two sets: train set and validation set with ratio 80:20. The train set used to update parameters and validation set used to tune hyperparameters.
  
For details, please have a look at the notebook which implement machine learning models and results.
