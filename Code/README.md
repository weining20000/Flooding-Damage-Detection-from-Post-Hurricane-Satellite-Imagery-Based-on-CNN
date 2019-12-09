# Instruction

This code folder contains all source code for this project. It is suggested to review the source code in the following order:

1. Pretrained_Models.py
      * Machine learning using pre-trained models (VGG-16 and Resnet50).
      * Since the pre-trained architectures are extremely large, it takes approximately two hours for the program to complete. It is recommended to set the number of training epochs, which is denoted by "N_EPOCHS", to a smaller number before you run the code. 
2. Custom_Model.py
      * Machine learning using custom convolutional neural network.
      * It takes approximately 15 minutes to run the code. 
3. satellite-images-of-hurricane-damage.zip
      + The dataset:
          i. train_another: the training data; 5000 images of each class
          ii. validation_another: the validation data; 1000 images of each class
          iii. test_another: the unbalanced test data; 8000/1000 images of damaged/undamaged classes
          vi. test: the balanced test data; 1000 images of each class (we only tested on the unbalanced dataset - test_another.)
