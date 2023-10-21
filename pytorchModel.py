import torch 
import cv2
import os
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from MonModule import *
from torchvision import transforms

def convert_tensor(img):
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    return transform(img)
  
width = 224
height = 224

pokemonDataDir = 'Data/pokemon'
pokemonFiles = [f for f in os.listdir(pokemonDataDir) if os.path.isfile(os.path.join(pokemonDataDir, f))]
digiDataDir = 'Data/digimon'
digimonFiles = [f for f in os.listdir(digiDataDir) if os.path.isfile(os.path.join(digiDataDir, f))]
OtherDataDir = 'Data/other'
OtherFiles = [f for f in os.listdir(OtherDataDir) if os.path.isfile(os.path.join(OtherDataDir, f))]
trainData = []
labelData = []

for filePath in pokemonFiles:
    img = cv2.imread(pokemonDataDir + '/'+ filePath) 
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = convert_tensor(img)  
    trainData.append(img)
    labelData.append(torch.FloatTensor([1,0,0]))

for filePath in digimonFiles:
    img = cv2.imread(digiDataDir + '/'+ filePath) 
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = convert_tensor(img)   
    trainData.append(img)
    labelData.append(torch.FloatTensor([0,1,0]))

for filePath in OtherFiles:
    img = cv2.imread(OtherDataDir + '/'+ filePath) 
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = convert_tensor(img)   
    trainData.append(img)
    labelData.append(torch.FloatTensor([0,0,1]))

dataset = MonDataSet(trainData, labelData)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


criterion = nn.CrossEntropyLoss()
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times
    print(epoch)
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data 
  
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


print('Finished Training')
  
torch.save(net.state_dict(), './mon_net.pth')
print('Finished Saving')

