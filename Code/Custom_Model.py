# Pytoch
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Data science tools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt


#=======================================================================================================================
#                                                   Load Data
#=======================================================================================================================
# Input images normalized in the same way and the image H and W are expected to be at least 224
# Reference: https://pytorch.org/docs/stable/torchvision/models.html
tforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                             transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
train_tfroms = transforms.Compose([transforms.Resize((128, 128)),transforms.ColorJitter(),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                   transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# get image data
os.system("sudo wget https://github.com/weining20000/Final-Project_Group4/blob/master/Code/satellite-images-of-hurricane-damage.zip")
os.system("sudo unzip satellite-images-of-hurricane-damage.zip")


# Load train image data
traindataFromFolders = datasets.ImageFolder(root = 'train_another/', transform = train_tfroms)
train_loader = DataLoader(traindataFromFolders, batch_size = 100,  shuffle = True)
x_train, y_train = iter(train_loader).next()

# Load validation image data
valdataFromFolders = datasets.ImageFolder(root = 'validation_another/', transform = tforms)
val_loader = DataLoader(valdataFromFolders,batch_size = 100, shuffle = True)
x_val, y_val = iter(val_loader).next()

# Load test image data
testdataFromFolders = datasets.ImageFolder(root = 'test_another/', transform = train_tfroms)
test_loader = DataLoader(testdataFromFolders,batch_size = 20, shuffle = False)
x_test, y_test = iter(test_loader).next()


#=======================================================================================================================
#                                                     Modeling
#=======================================================================================================================
# %% ------------------------------------------------- Set-Up ----------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %% ---------------------------------------------- Hyper Parameters ---------------------------------------------------
LR = 0.01
N_EPOCHS = 100
BATCH_SIZE = 25
DROPOUT = 0.5


# %% ---------------------------------------------- Helper Functions ---------------------------------------------------
def acc(x, y, return_labels = False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------------------- CNN Class-------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(64, 64, kernel_size = 6, stride = 1, padding = 1)
        self.convnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d((2, 2))

        self.dropout = nn.Dropout(DROPOUT)
        self.linear1 = nn.Linear(64 * 13 * 13, 16)
        self.linear1_bn = nn.BatchNorm1d(16)
        self.linear2 = nn.Linear(16, 2)
        self.linear2_bn = nn.BatchNorm1d(2)
        self.sigmoid = torch.sigmoid
        self.relu = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.relu(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.relu(self.conv2(x))))
        x = self.pool3(self.convnorm3(self.relu(self.conv3(x))))
        # print(x.shape)
        x = self.dropout(self.linear1_bn(self.relu(self.linear1(x.view(-1, 64 * 13 * 13)))))
        x = self.dropout(self.linear2_bn(self.relu(self.linear2(x))))
        x = self.sigmoid(x)
        return x


# %% ------------------------------------------------- Training Prep ---------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum = 0.9)
criterion = nn.CrossEntropyLoss()

def acc(x, y, return_labels = False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis = 1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------------------- Training Loop --------------------------------------------------
print("Starting training loop...")
history_li = []
for epoch in range(N_EPOCHS):

    # keep track of training and validation loss each epoch
    train_loss = 0.0
    val_loss = 0.0

    train_acc = 0
    val_acc = 0

    # Set to training
    model.train()
    start = timer()

    loss_train = 0
    model.train()

    for batch in range(len(x_train)//BATCH_SIZE):

        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

        # Track train loss
        train_loss += loss.item()
        train_acc = acc(x_train, y_train)

    model.eval()

    with torch.no_grad():
        y_val_pred = model(x_val)
        loss = criterion(y_val_pred, y_val)
        val_loss = loss.item()
        val_acc = acc(x_val, y_val)
        loss_test = loss.item()

        history_li.append([train_loss/BATCH_SIZE, val_loss, train_acc, val_acc])
        torch.save(model.state_dict(), 'model_custom.pt')
        torch.cuda.empty_cache()
    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train/BATCH_SIZE, acc(x_train, y_train), val_loss, acc(x_val, y_val)))

    history = pd.DataFrame(history_li, columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])


#=======================================================================================================================
#                                                   Visualization
#=======================================================================================================================
history.to_csv("custom_result.csv")


df_valid_loss = pd.DataFrame({'Epoch': range(0, 100), # Make sure the range is consistent with the Epoch number
                       'valid_loss_train':history['train_loss'],
                       'valid_loss_val': history['val_loss']
                       })

plot1, = plt.plot('Epoch', 'valid_loss_train', data = df_valid_loss, color = 'skyblue')
plot2, = plt.plot('Epoch', 'valid_loss_val', data = df_valid_loss, linestyle = '--', color = 'orange')
plt.xlabel('Epoch')
plt.ylabel('Average Validation Loss per Batch')
plt.title('Model Custom: Training and Validation Loss', pad = 20)
plt.legend([plot1, plot2], ['training loss', 'validation loss'])
plt.savefig('Result_Loss_Custom.png')


#=======================================================================================================================
#                                                      Prediction
#=======================================================================================================================

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

y_actual, y_predict = predict(model, "model_custom.pt", test_loader)


#=======================================================================================================================
#                                                       Evaluation
#=======================================================================================================================
acc_rate = 100*accuracy_score(y_actual, y_predict)
print("The Accuracy rate for the model is: ", acc_rate)
print(confusion_matrix(y_actual, y_predict))

cm = confusion_matrix(y_actual, y_predict)
fig = plt.figure(figsize = (10,7))
ax= plt.subplot()
sns.heatmap(cm, cmap="Blues", annot=True, ax = ax, fmt='g', annot_kws={"size": 30})

# labels, title and ticks
ax.set_xlabel('Predicted labels',fontsize= 20)
ax.set_ylabel('True labels',fontsize= 20)

ax.set_title('Custom Model \n',fontsize= 28)

ax.xaxis.set_ticklabels(['damage', 'no damage'],fontsize= 20)
ax.yaxis.set_ticklabels(['damage', 'no damage'],fontsize= 20)
ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))

fig.savefig("Result_Confusion_Matrix_Custom.png")
