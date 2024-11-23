import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import os
import torch.nn as nn
import torch.optim as optim


SCALE_FACTOR = .25


class ImageDataset(Dataset):
    def __init__(self, inputs, labels, transform=None, target_transform=None):
        self.img_labels = labels
        self.imgs = inputs
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            v2.RandomHorizontalFlip(p=0.5),
        ])
        self.target_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label

train = "./dataset/MineCraft-RT_1280x720_v14/MineCraft-RT_1280x720_v14/images"
print("Loading Training Data")
labels = []
inputs = []
for dirname, _, filenames in os.walk(train):
    for filename in filenames:
        label = cv2.imread(dirname +'/'+ filename)
        input = cv2.resize(label, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR) 
        labels.append(label)
        inputs.append(input)
train_dataset = ImageDataset(inputs,labels)    
       
test = "./labels/MineCraft-RT_1280x720_v12/MineCraft-RT_1280x720_v12"
print("Loading Test Data")
labels = []
inputs = []
for dirname, _, filenames in os.walk(test + '/images'):
    for filename in filenames:
        label = cv2.imread(dirname +'/'+ filename)
        input = cv2.resize(label, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR) 
        labels.append(label)
        inputs.append(input)
test_dataset = ImageDataset(inputs,labels)    
print(train_dataset.__getitem__(1)[0].shape,train_dataset.__getitem__(1)[1].shape)

train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True, num_workers=0)
test_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True, num_workers=0)

###THIS IS UP TO THE POINT I ACTUALLY VERIFIED THINGS WORK

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        #Upsampling layer
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # Patch extraction and representation
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        # Non-linear mapping
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        # Reconstruction
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        # Activation function (ReLU)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.upsample(x)  # Upsample before feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.res_scale = 0.1  # Residual scaling factor

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + self.res_scale * residual

class EDSR(nn.Module):
    def __init__(self, n_resblocks=8, n_feats=64, scale_factor=4):
        super(EDSR, self).__init__()
        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_resblocks)])
        
        # Intermediate convolution
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        
        # Upsampling layer
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        
        # Final output layer
        self.conv3 = nn.Conv2d(n_feats, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        
        # Pass through residual blocks
        x = self.res_blocks(x)
        x = self.conv2(x)
        x += residual  # Skip connection from the initial feature extraction
        
        # Upsample to high resolution
        x = self.upsample(x)
        x = self.conv3(x)
        return x

#initialize the model, loss function, and optimizer
model = EDSR()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)

    #validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(test_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}],Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
#save the trained model
torch.save(model.state_dict(), 'srcnn_model.pth')
print("Model saved as 'srcnn_model.pth'")