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

testDataDir = 'Data/testData'
testDataFiles = [f for f in os.listdir(testDataDir) if os.path.isfile(os.path.join(testDataDir, f))] 
 
net = Net()
net.load_state_dict(torch.load('./mon_net.pth'))

testData = []
labelData = []

for filePath in testDataFiles:
    img = cv2.imread(testDataDir + '/'+ filePath) 
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = convert_tensor(img)   
    testData.append(img)

    monType = filePath.split('_')[0]

    if monType == 'pokemon':
        labelData.append(torch.FloatTensor([1,0,0]))
    elif monType == 'digimon':
        labelData.append(torch.FloatTensor([0,1,0]))
    elif monType == 'other':
        labelData.append(torch.FloatTensor([0,0,1]))

dataset = MonDataSet(testData, labelData)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

totalCnt = 0
correctCnt = 0
for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data 
  
        result = net(inputs)
        _, predicted = torch.max(result, 1)
        _, expect = torch.max(labels, 1)

        preName = ''
        if predicted.item() == 0:
            preName = 'pokemon'
        elif predicted.item() == 1:
             preName = 'digimon'
        elif predicted.item() == 2:
             preName = 'other'

        expectName = ''
        if expect.item() == 0:
            expectName = 'pokemon'
        elif expect.item() == 1:
             expectName = 'digimon'
        elif expect.item() == 2:
             expectName = 'other'

        if expectName == preName:
             correctCnt +=1

        totalCnt += 1 
        print('Expection:' + expectName + '  Predict:' + preName)

print('Correct Rate:' + str(correctCnt/totalCnt * 100) + '%')
 