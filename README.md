# Final-Project_Group4

## Abstract

In this project, we design and train a convolutional neural network from scratch and compare it with two existing neural networks, the VGG-16 and Resnet50, for a binary classification problem. The dataset we use in this project is a collection of satellite images that cover the Greater Houston area before and after the Hurricane Harvey in 2017. Our goal is to use convolutional neural networks to differentiate images that contain damages (i.e. flooding area or damaged buildings) from the ones still intact. Our research results show that a custom model with fewer convolutional layers performs better on the hold-out test set than the other two complicated pre-trained models. Our best model is able to achieve an accuracy of over 90% on the hold-out test set. We train our networks using the PyTorch framework with a single NVIDIA K80 Tesla GPU.      
