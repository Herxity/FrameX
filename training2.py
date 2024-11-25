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
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt


SCALE_FACTOR = .5

class ImageDataset(Dataset):
    def __init__(self, labels, dirname, patch_size=256, transform=None, target_transform=None):
        self.img_labels = labels
        self.dirname = dirname
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load the full image
        filename = self.img_labels[idx]
        image = cv2.imread(os.path.join(self.dirname, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Downscale for input (low-res) and upscale back for comparison (high-res)
        h, w, _ = image.shape
        scale_factor = 2  # Adjust based on your scaling requirement
        lr_image = cv2.resize(image, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, (w, h), interpolation=cv2.INTER_CUBIC)

        # Extract a random patch
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr_patch = image[y:y + self.patch_size, x:x + self.patch_size, :]
        lr_patch = lr_image[y:(y + self.patch_size), x:(x + self.patch_size), :]
        
        # Apply transforms
        if self.transform:
            lr_patch = self.transform(lr_patch)
        if self.target_transform:
            hr_patch = self.target_transform(hr_patch)
        # Convert to PyTorch tensors if not already
        lr_patch = torch.from_numpy(lr_patch)
        hr_patch = torch.from_numpy(hr_patch)

        # Make channels first
        lr_patch = lr_patch.permute(2, 0, 1)
        hr_patch = hr_patch.permute(2, 0, 1)
        return lr_patch, hr_patch

    # Method to disassemble the image into patches
    def extract_patches(self, image, patch_size=256):
        h, w, _ = image.shape
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size, :]
                patches.append(patch)
        return patches
    # Method to reassemble the patches into the original image
    def reassemble_image(self, patches, image_size=(1080, 1920)):
        h, w = image_size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        idx = 0
        patch_size = patches[0].shape[0]
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = patches[idx]
                image[i:i + patch_size, j:j + patch_size, :] = patch
                idx += 1
        return image

images_dir = "./MineCraft-RT_1280x720_v14/MineCraft-RT_1280x720_v14/images/"
print("Loading Image Data")
image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Split the dataset into training and testing sets
train_filenames, test_filenames = train_test_split(image_filenames, test_size=0.2, random_state=42)

train_dataset = ImageDataset(train_filenames, images_dir,patch_size=96)
test_dataset = ImageDataset(test_filenames, images_dir,patch_size=96)

print(train_dataset.__getitem__(1)[0].shape, train_dataset.__getitem__(1)[1].shape)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)

###THIS IS UP TO THE POINT I ACTUALLY VERIFIED THINGS WORK


# Test out grabbing an image from the dataloader
sample_batch = next(iter(train_dataloader))
sample_input, sample_target = sample_batch
print(sample_input.shape, sample_target.shape)
#Save the image
sample_input = sample_input[0].permute(1, 2, 0).numpy()
sample_target = sample_target[0].permute(1, 2, 0).numpy()

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Upsampling layer
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
    def __init__(self, n_resblocks=12, n_feats=128, scale_factor=2):
        super(EDSR, self).__init__()
        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_resblocks)])
        
        # Intermediate convolution
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        
        # # Upsampling layer
        # self.upsample = nn.Sequential(
        #     nn.Conv2d(n_feats, n_feats * (scale_factor ** 2), kernel_size=3, padding=1),
        #     nn.PixelShuffle(scale_factor)
        # )
        
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
        # x = self.upsample(x)
        x = self.conv3(x)
        return x

print("Initializing model...")
# Initialize the model, loss function, and optimizer
model = EDSR()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
num_epochs = 30

#Calculate PSNR and SSIM
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))





for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()  # Initialize scaler
    for i, (inputs, labels) in tqdm(enumerate(train_dataloader)):
        
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        
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

    # Validation loop
    model.eval()
    val_loss = 0.0
    all_inputs = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # Explicitly delete variables
            del inputs, labels, outputs
            
    # Pick a random image from the dataloader and run inference
    random_batch = next(iter(test_dataloader))
    random_idx = random.randint(0, random_batch[0].size(0) - 1)
    random_input = random_batch[0][random_idx].unsqueeze(0).to(device).float()
    random_target = random_batch[1][random_idx].unsqueeze(0).to(device).float()

    with torch.no_grad():
        random_output = model(random_input)

    # Move the tensors to CPU and convert to numpy arrays
    random_input = random_input.cpu().squeeze(0).numpy().transpose(1, 2, 0)
    random_output = random_output.cpu().squeeze(0).numpy().transpose(1, 2, 0)
    random_target = random_target.cpu().squeeze(0).numpy().transpose(1, 2, 0)


    # # Denormalize the images
    # random_input = (random_input * 0.5 + 0.5) * 255.0
    # random_output = (random_output * 0.5 + 0.5) * 255.0

    # Save the images
    epoch_dir = f'/epoch_{epoch+1}'
    os.makedirs(f'./runs/{epoch_dir}', exist_ok=True)
    plt.imsave(f'runs/{epoch_dir}/input_{random_idx}.png', random_input.astype(np.uint8))
    plt.imsave(f'runs/{epoch_dir}/output_{random_idx}.png', random_output.astype(np.uint8))
    plt.imsave(f'runs/{epoch_dir}/output_{random_idx}.png', random_target.astype(np.uint8))

    
    avg_val_loss = val_loss / len(test_dataloader)
    # Calculate PSNR and SSIM
    psnr_value = psnr(random_output, random_target)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, PSNR: {psnr_value:.4f}')
    
    if epoch % 5 == 0:
        # Save the trained model
        torch.save(model.state_dict(), f'srcnn_model-{epoch}-{avg_val_loss:.4f}.pth')
        print(f"Model checkpoint saved as 'srcnn_model-{epoch}-{avg_val_loss:.4f}.pth'")
    torch.cuda.empty_cache()
    
# Save the final model
torch.save(model.state_dict(), 'srcnn_model.pth')
print("Model saved as 'srcnn_model.pth'")

# Get patches, infer, reassemble. Take from imagedataset
patches = test_dataset.extract_patches(sample_input)
inference_patches = []
model.eval()
with torch.no_grad():
    for patch in patches:
        patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device).float()
        output = model(patch)
        output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        inference_patches.append(output)

inference_image = test_dataset.reassemble_image(inference_patches, sample_input.shape[:2])
plt.imsave('inference_image.png', inference_image)