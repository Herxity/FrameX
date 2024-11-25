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
from tqdm import tqdm


SCALE_FACTOR = .5


class ImageDataset(Dataset):
    def __init__(self, labels,dirname, transform=None, target_transform=None):
        self.img_labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize images
            v2.RandomHorizontalFlip(p=0.5)
        ])

        self.target_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        file = self.img_labels[idx]
        label = cv2.imread(dirname +'/'+ filename)
        image = cv2.resize(label, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR) 

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label

train = "./dataset/Train/"
print("Loading Training Data")
train_labels = []
for dirname, _, filenames in os.walk(train):
    for filename in filenames:
        train_labels.append(dirname)
train_dataset = ImageDataset(train_labels,train)    
       
test = "./dataset/Test/"
print("Loading Test Data")
test_labels = []
for dirname, _, filenames in os.walk(test):
    for filename in filenames:
        test_labels.append(dirname)

test_dataset = ImageDataset(test_labels,test)    
print(train_dataset.__getitem__(1)[0].shape,train_dataset.__getitem__(1)[1].shape)

train_dataloader = DataLoader(train_dataset, batch_size=16,shuffle=True, num_workers=8,pin_memory=True,prefetch_factor =2)
test_dataloader = DataLoader(test_dataset, batch_size=16,shuffle=True, num_workers=8,pin_memory=True,prefetch_factor =2)

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
    def __init__(self, n_resblocks=2, n_feats=64, scale_factor=2):
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

print("Initializing model...")
#initialize the model, loss function, and optimizer
model = EDSR()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()  # Initialize scaler
    for i, (inputs, labels) in tqdm(enumerate(train_dataloader)):
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Automatic mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        del inputs, labels, outputs
        torch.cuda.empty_cache()

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
            # Explicitly delete variables
            del inputs, labels, outputs
    avg_val_loss = val_loss / len(test_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}],Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    
    if epoch % 5 == 0:
        #save the trained model
        torch.save(model.state_dict(), f'srcnn_model-{epoch}-{avg_val_loss:.4f}.pth')
        print(f"Model checkpoint saved as 'srcnn_model-{epoch}-{avg_val_loss:.4f}.pth'")
    torch.cuda.empty_cache()