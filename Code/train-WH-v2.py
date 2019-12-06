# Pytoch
import torch
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Data science tools
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Image manipulations
from PIL import Image
# Useful for examining network
#pip install torchsummary
#from torchsummary import summary
#os.system("sudo pip install torchsummary")
# Reference: https://github.com/sksq96/pytorch-summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns


#==========================================================================
#                                Load Data
#==========================================================================
# Input images normalized in the same way and the image H and W are expected to be at least 224
# Reference: https://pytorch.org/docs/stable/torchvision/models.html
tforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_tfroms = transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load train data
# traindataFromFolders = datasets.ImageFolder(root='small_train/train_another/', transform=train_tfroms)
# train_loader = DataLoader(traindataFromFolders, batch_size=1, shuffle=True)

traindataFromFolders = datasets.ImageFolder(root='Metadata/train_another/', transform=train_tfroms)
train_loader = DataLoader(traindataFromFolders, batch_size=50, shuffle=True)
x_train, y_train = iter(train_loader).next()

# Load validation data
# valdataFromFolders = datasets.ImageFolder(root='small_train/validation_another/', transform=tforms)
# val_loader = DataLoader(valdataFromFolders, batch_size=1, shuffle=True)

valdataFromFolders = datasets.ImageFolder(root='Metadata/validation_another/', transform=tforms)
val_loader = DataLoader(valdataFromFolders, batch_size=50, shuffle=True)
x_val, y_val = iter(val_loader).next()

#transforms.ToPILImage()(x_val[101]).show()


#==========================================================================
#                             Modeling
#==========================================================================
# %% ------------------------- Set-Up ------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load pre-trained model
def get_pretrained_model(model_name):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features
        n_classes = 2

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes))

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        n_classes = 2
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes))

    # Move to GPU
    MODEL = model.to(device)

    return MODEL

#----------------------------------Set Up varied models---------------------------------------
# VGG 16
model_vgg = get_pretrained_model('vgg16')
criterion_vgg = nn.CrossEntropyLoss()
optimizer_vgg = torch.optim.Adam(model_vgg.parameters(), lr=0.001)

# ResNet 50
model_resnet50 = get_pretrained_model('resnet50')
criterion_resnet50 = nn.CrossEntropyLoss()
optimizer_resnet50 = torch.optim.Adam(model_resnet50.parameters(), lr=0.001)


#---------------------------Helper function to calculate accuracy rate------------------------
def acc_vgg(x, y, return_labels=False):
    with torch.no_grad():
        logits = model_vgg(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)

def acc_resnet50(x, y, return_labels=False):
    with torch.no_grad():
        logits = model_resnet50(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)

#------------------------------------Training Function----------------------------------------
def train(model, criterion, optimizer, acc, xtrain, ytrain, xval, yval, save_file_name, n_epochs, BATCH_SIZE):

    history1 = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        val_loss = 0.0

        train_acc = 0
        val_acc = 0

        # Set to training
        model.train()
        start = timer()

        #Training loop
        for batch in range(len(xtrain)//BATCH_SIZE + 1):
            idx = slice(batch * BATCH_SIZE, (batch+1)*BATCH_SIZE)

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs
            output = model(xtrain[idx])
            # Loss and BP of gradients
            loss = criterion(output, ytrain[idx])
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Track train loss
            train_loss += loss.item()
            train_acc = acc(xtrain, ytrain)

        # After training loops ends, start validation
        # set to evaluation mode
        model.eval()
        # Don't need to keep track of gradients
        with torch.no_grad():
            # Evaluation loop
            # F.P.
            y_val_pred = model(xval)
            # Validation loss
            loss = criterion(y_val_pred, yval)
            val_loss = loss.item()
            val_acc = acc(xval, yval)

            history1.append([train_loss, val_loss, train_acc, val_acc])
            torch.save(model.state_dict(), save_file_name)
            torch.cuda.empty_cache() # reference: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530

            # Print training and validation results
        print("Epoch {} | Train Loss: {:.5f} | Train Acc: {:.2f} | Test Loss: {:.5f} | Test Acc: {:.2f} |".format(
            epoch, train_loss / BATCH_SIZE, acc(xtrain, ytrain), val_loss, acc(xval, yval)))
        # Format history
        history = pd.DataFrame(history1, columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])
    return model, history

model_vgg, history_vgg = train(model_vgg,
                       criterion_vgg,
                       optimizer_vgg,
                       acc_vgg,
                       x_train,
                       y_train,
                       x_val,
                       y_val,
                       save_file_name = 'model_vgg.pt',
                       n_epochs=30, # if change epoch number, don't forget to change the range limits in the visualization below
                       BATCH_SIZE = 4)

model_resnet50, history_resnet50 = train(model_resnet50,
                       criterion_resnet50,
                       optimizer_resnet50,
                       acc_resnet50,
                       x_train,
                       y_train,
                       x_val,
                       y_val,
                       save_file_name = 'model_resnet50.pt',
                       n_epochs=30, # if change epoch number, don't forget to change the range limits in the visualization below
                       BATCH_SIZE = 4)

#==========================================================================
#                                Visualization
#==========================================================================

df_valid_loss = pd.DataFrame({'Epoch': range(0, 30), # Make sure the range is consistent with the Epoch number
                       'valid_loss_vgg': history_vgg['val_loss'],
                       'valid_loss_resnet50':history_resnet50['val_loss']
                       })
plot1, = plt.plot('Epoch', 'valid_loss_vgg', data=df_valid_loss, linestyle = '--', color = 'skyblue')
plot2, = plt.plot('Epoch', 'valid_loss_resnet50', data=df_valid_loss, color = 'orange')
plt.xlabel('Epoch')
plt.ylabel('Average Validation Loss per Batch')
plt.title('Validation Losses Comparison between VGG16 and Resnet50', pad =20)
plt.legend([plot1, plot2], ['VGG16', 'Resnet50'])
plt.savefig('Result_Comparison.png')
# Reference: https://python-graph-gallery.com/122-multiple-lines-chart/


#==========================================================================
#                                Prediction
#==========================================================================
testdataFromFolders = datasets.ImageFolder(root='Metadata/test_another/', transform=train_tfroms)
test_loader = DataLoader(testdataFromFolders, batch_size= 20, shuffle=False)

def predict(mymodel, model_name_pt, loader):

    model = mymodel
    model.load_state_dict(torch.load(model_name_pt))
    model.to(device)
    model.eval()
    y_actual_np = []
    y_pred_np = []
    for idx, data in enumerate(test_loader):
        test_x, test_label = data[0], data[1]
        test_x = test_x.to(device)
        y_actual_np.extend(test_label.cpu().numpy().tolist())

        with torch.no_grad():
            y_pred_logits = model(test_x)
            pred_labels = np.argmax(y_pred_logits.cpu().numpy(), axis=1)
            print("Predicting ---->", pred_labels)
            y_pred_np.extend(pred_labels.tolist())

    return y_actual_np, y_pred_np

y_actual, y_predict_vgg = predict(model_vgg, "model_vgg.pt", test_loader)

#==========================================================================
#                                Evaluation
#==========================================================================
acc_rate_vgg = 100*accuracy_score(y_actual, y_predict_vgg)
print("The Accuracy rate for the VGG-16 model is: ", acc_rate_vgg)
# Confusion matrix for model-VGG-16
print(confusion_matrix(y_actual, y_predict_vgg))
# Reference: other performance matrixes https://medium.com/hugo-ferreiras-blog/confusion-matrix-and-other-metrics-in-machine-learning-894688cb1c0a











