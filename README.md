# Flooding Damage Detection from Post-Hurricane Satellite Imagery Based on Convolutional Neural Networks

## Requirements
* PyCharm
* PyTorch
* Custom CNN, VGG-16, Resnet50

## Abstract

In this project, we designed and trained a convolutional neural network from scratch and compared it with two existing neural networks, the VGG-16 and Resnet50, for a binary classification problem. The dataset we used in this project is a collection of satellite images that cover the Greater Houston area before and after the Hurricane Harvey in 2017. The goal was to use convolutional neural networks to differentiate images that contain damages (i.e. flooding area or damaged buildings) from the ones still intact. Our research results show that a custom model with fewer convolutional layers performs better on the hold-out test set than the other two complicated pre-trained models. Our best model was able to achieve an accuracy of over 90% on the hold-out test set. This project trained the networks using PyTorch as the framework with a single NVIDIA K80 Tesla GPU. (The full report is attached as a pdf in the **Final-Group-Project-Report** repository.)      
