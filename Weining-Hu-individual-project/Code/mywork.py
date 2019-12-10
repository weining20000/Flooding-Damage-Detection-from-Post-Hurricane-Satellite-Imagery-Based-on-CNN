# Pytoch
import torch
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch_lr_finder import LRFinder

# Data science tools
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Image manipulations
from PIL import Image
# Useful for examining network
# pip install torchsummary
# from torchsummary import summary
#os.system("sudo pip install torchsummary")
#os.system("sudo pip install torch-lr-finder")
# Reference: https://github.com/sksq96/pytorch-summary

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

os.system("sudo wget https://storage.googleapis.com/group4finalproject/satellite-images-of-hurricane-damage.zip")
os.system("sudo unzip satellite-images-of-hurricane-damage.zip")


#==========================================================================
#                                Load Data
#==========================================================================
tforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_tfroms = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

# Input images normalized in the same way and the image H and W are expected to be at least 224
# Reference: https://pytorch.org/docs/stable/torchvision/models.html

# Load train data
traindataFromFolders = datasets.ImageFolder(root='train_another/', transform=train_tfroms)
train_loader = DataLoader(traindataFromFolders, batch_size=50, shuffle=True)
x_train, y_train = iter(train_loader).next()

# Load validation data
valdataFromFolders = datasets.ImageFolder(root='validation_another/', transform=tforms)
val_loader = DataLoader(valdataFromFolders, batch_size=50, shuffle=True)
x_val, y_val = iter(val_loader).next()

# Load test data
testdataFromFolders = datasets.ImageFolder(root='test_another/', transform=train_tfroms)
test_loader = DataLoader(testdataFromFolders, batch_size = 40, shuffle=False)

#==========================================================================
#                             Modeling
#==========================================================================
# %% ------------------------- Set-Up ------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Load pre-trained model
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
optimizer_vgg = torch.optim.Adam(model_vgg.parameters(), lr=0.00002)

# ResNet 50
model_resnet50 = get_pretrained_model('resnet50')
criterion_resnet50 = nn.CrossEntropyLoss() #nn.BCELoss()
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

        #Training loop
        for batch in range(len(xtrain)//BATCH_SIZE):
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

            history1.append([train_loss / BATCH_SIZE, val_loss, train_acc, val_acc])
            torch.save(model.state_dict(), save_file_name)
            torch.cuda.empty_cache()

            # Print training and validation results
        print("Epoch {} | Train Loss: {:.5f} | Train Acc: {:.2f} | Valid Loss: {:.5f} | Valid Acc: {:.2f} |".format(
            epoch, train_loss / BATCH_SIZE, acc(xtrain, ytrain), val_loss, acc(xval, yval)))
        # Format history
        history = pd.DataFrame(history1, columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])
    return model, history

N_EPOCHS = 30
model_vgg, history_vgg = train(model_vgg,
                       criterion_vgg,
                       optimizer_vgg,
                       acc_vgg,
                       x_train,
                       y_train,
                       x_val,
                       y_val,
                       save_file_name = 'model_vgg.pt',
                       n_epochs = N_EPOCHS,
                       BATCH_SIZE = 3)

model_resnet50, history_resnet50 = train(model_resnet50,
                       criterion_resnet50,
                       optimizer_resnet50,
                       acc_resnet50,
                       x_train,
                       y_train,
                       x_val,
                       y_val,
                       save_file_name = 'model_resnet50.pt',
                       n_epochs = N_EPOCHS,
                       BATCH_SIZE = 4)

#==========================================================================
#                                Visualization
#==========================================================================
#-----------------------------------Plot 1---------------------------------
plt.figure()
vgg_train_acc = history_vgg['train_acc']
vgg_val_acc = history_vgg['val_acc']
vgg_epoch = range(0, len(vgg_train_acc), 1)
plot1, = plt.plot(vgg_epoch, vgg_train_acc, linestyle = "solid", color = "skyblue")
plot2, = plt.plot(vgg_epoch, vgg_val_acc, linestyle = "dashed", color = "orange")
plt.legend([plot1, plot2], ['training acc', 'validation acc'])
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy per Batch')
plt.title('Model VGG-16: Training and Validation Accuracy', pad = 20)
plt.savefig('VGG16-Acc-Plot.png')

plt.figure()
vgg_train_loss = history_vgg['train_loss']
vgg_val_loss = history_vgg['val_loss']
vgg_epoch = range(0, len(vgg_train_loss), 1)
plot3, = plt.plot(vgg_epoch, vgg_train_loss, linestyle = "solid", color = "skyblue")
plot4, = plt.plot(vgg_epoch, vgg_val_loss, linestyle = "dashed", color = "orange")
plt.legend([plot3, plot4], ['training loss', 'validation loss'])
plt.xlabel('Epoch')
plt.ylabel('Average Loss per Batch')
plt.title('Model VGG-16: Training and Validation Loss', pad = 20)
plt.savefig('VGG16-Loss-Plot.png')

#-----------------------------------Plot 2---------------------------------
# Training Reseults: Resnet50
plt.figure()
resnet50_train_acc = history_resnet50['train_acc']
resnet50_val_acc = history_resnet50['val_acc']
resnet50_epoch = range(0, len(resnet50_train_acc), 1)
plot5, = plt.plot(resnet50_epoch, resnet50_train_acc, linestyle = "solid", color = "skyblue")
plot6, = plt.plot(resnet50_epoch, resnet50_val_acc, linestyle = "dashed", color = "orange")
plt.legend([plot5, plot6], ['training acc', 'validation acc'])
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy per Batch')
plt.title('Model Resnet50: Training and Validation Accuracy', pad = 20)
plt.savefig('Resnet50-Acc-Plot.png')

plt.figure()
resnet50_train_loss = history_resnet50['train_loss']
resnet50_val_loss = history_resnet50['val_loss']
resnet50_epoch = range(0, len(resnet50_train_loss), 1)
plot7, = plt.plot(resnet50_epoch, resnet50_train_loss, linestyle = "solid", color = "skyblue")
plot8, = plt.plot(resnet50_epoch, resnet50_val_loss, linestyle = "dashed", color = "orange")
plt.legend([plot7, plot8], ['training loss', 'validation loss'])
plt.xlabel('Epoch')
plt.ylabel('Average Loss per Batch')
plt.title('Model Resnet50: Training and Validation Loss', pad = 20)
plt.savefig('Resnet50-Loss-Plot.png')

#-----------------------------------Plot 3---------------------------------

plt.figure()
df_valid_loss = pd.DataFrame({'Epoch': range(0, N_EPOCHS, 1),
                       'valid_loss_vgg': history_vgg['val_loss'],
                       'valid_loss_resnet50':history_resnet50['val_loss']
                       })
plota, = plt.plot('Epoch', 'valid_loss_vgg', data=df_valid_loss, linestyle = '--', color = 'skyblue')
plotb, = plt.plot('Epoch', 'valid_loss_resnet50', data=df_valid_loss, color = 'orange')
plt.xlabel('Epoch')
plt.ylabel('Average Validation Loss per Batch')
plt.title('Validation Loss Comparison', pad = 20)
plt.legend([plota, plotb], ['VGG16', 'Resnet50'])
plt.savefig('Result_Comparison.png')

#==========================================================================
#                                Prediction
#==========================================================================
def predict(mymodel, model_name_pt, loader):

    model = mymodel
    model.load_state_dict(torch.load(model_name_pt))
    model.to(device)
    model.eval()
    y_actual_np = []
    y_pred_np = []
    for idx, data in enumerate(loader):
        test_x, test_label = data[0], data[1]
        test_x = test_x.to(device)
        y_actual_np.extend(test_label.cpu().numpy().tolist())

        with torch.no_grad():
            y_pred_logits = model(test_x)
            pred_labels = np.argmax(y_pred_logits.cpu().numpy(), axis=1)
            print("Predicting ---->", pred_labels)
            y_pred_np.extend(pred_labels.tolist())

    return y_actual_np, y_pred_np

y_actual_vgg, y_predict_vgg = predict(model_vgg, "model_vgg.pt", test_loader)
y_actual_resnet50, y_predict_resnet50 = predict(model_resnet50, "model_resnet50.pt", test_loader)


#==========================================================================
#                                Evaluation
#==========================================================================
# VGG-16 Accuracy
print("=====================================================")
acc_rate_vgg = 100*accuracy_score(y_actual_vgg, y_predict_vgg)
print("The Accuracy rate for the VGG-16 model is: ", acc_rate_vgg)
# Confusion matrix for model-VGG-16
print("The Confusion Matrix for VGG-16 is as below:")
print(confusion_matrix(y_actual_vgg, y_predict_vgg))

# Resnet50 Accuracy
print("=====================================================")
acc_rate_resnet50 = 100*accuracy_score(y_actual_resnet50, y_predict_resnet50)
print("The Accuracy rate for the Resnet50 model is: ", acc_rate_resnet50)
# Confusion matrix for model Resnet50
print("The Confusion Matrix for Resnet50 is as below:")
print(confusion_matrix(y_actual_resnet50, y_predict_resnet50))







