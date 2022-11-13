import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy


transform_tensor_to_pil = ToPILImage()
transform_pil_to_tensor = ToTensor()

def load_data():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data
  
  
def create_dataloaders(training_data, test_data, batch_size=64):

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
        
    return train_dataloader, test_dataloader
  
def loss_fun(y_pred, y_ground):
  v = -(y_ground * torch.log(y_pred + 0.0001))
  v = torch.sum(v)
  return v

class cs19b045cnn(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=param[1][0], 
                               out_channels=param[1][1], 
                               kernel_size=param[1][2], 
                               stride=param[1][3], 
                               padding=param[1][4])
        self.conv2 = nn.Conv2d(in_channels=param[2][0], 
                               out_channels=param[2][1], 
                               kernel_size=param[2][2], 
                               stride=param[2][3], 
                               padding=param[2][4])
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.fc_nodes_calc(), out_features=num_classes)
        self.m = nn.Softmax(dim =1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.m(x)
        return x

    def fc_nodes_calc(self):
      new_w = param[0][1]
      new_h = param[0][2]
      for i in range(conv_layers):
        new_w = (new_w - param[i+1][2][0] + 1)/param[i+1][3]
        new_h = (new_h - param[i+1][2][1] + 1)/param[i+1][3]
      size = int(param[conv_layers][1] * new_w * new_h)
      return size
    
def get_model(train_loader,param,e = 10):
	model = cs19b045cnn(param)
	return model

def train_network2(train_loader, optimizer,criteria, e):
  for epoch in range(e): 

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        tmp = torch.nn.functional.one_hot(labels, num_classes= 10)
        loss = criteria(outputs, tmp)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

  print('Finished Training')
  
def test2(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            tmp = torch.nn.functional.one_hot(y, num_classes= 10)
            pred = model(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    accuracy1 = Accuracy()
    print('Accuracy :', accuracy1(pred,y))
    precision = Precision(average = 'macro', num_classes = 10)
    print('precision :', precision(pred,y))

    recall = Recall(average = 'macro', num_classes = 10)
    print('recall :', recall(pred,y))
    f1_score = F1Score(average = 'macro', num_classes = 10)
    print('f1_score :', f1_score(pred,y))
    return accuracy1,precision, recall, f1_score  
