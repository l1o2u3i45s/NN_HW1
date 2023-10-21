import torch   
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset 


class MonDataSet(Dataset):
    def __init__(self, image_list, label_list):
        self.image_list = image_list
        self.label_list = label_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.label_list[idx]
        return image, label

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输出尺寸: 224x224x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出尺寸: 112x112x32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 输出尺寸: 112x112x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出尺寸: 56x56x64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 输出尺寸: 56x56x128
            nn.ReLU(),
            nn.MaxPool2d(2)   # 输出尺寸: 28x28x128
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(28*28*128, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc_layers(x)
        return x