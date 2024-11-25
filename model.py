import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import cv2

class EDSR(nn.Module):
    def __init__(self, n_resblocks=8, n_feats=128, scale_factor=2):
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

class SRCNNWrapper:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EDSR().to(self.device)
        if model_path:
            self.load_model(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize images
        ])

    def load_model(self, model_path):
        """Load pre-trained model weights."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")

    def preprocess_image(self, image):
        """Preprocess the image for SRCNN."""
        image = ToTensor()(image).unsqueeze(0).to(self.device)  # Add batch dimension
        image = self.transform(image)
        
        return image

    def enhance_image(self, image_path):
        """Enhance the image resolution."""
        # Preprocess
        input_image = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            output_image = self.model(input_image)
        
        # Post-process
        output_image = output_image.squeeze(0).cpu()
        output_image = ToPILImage()(output_image)
        return output_image

    def save_image(self, image, output_path):
        """Save the enhanced image to disk."""
        image.save(output_path)
        print(f"Enhanced image saved to {output_path}")

    def display_image(self, image):
        """Display the enhanced image."""
        image.show()


# Example usage
if __name__ == "__main__":
    srcnn_wrapper = SRCNNWrapper(model_path="./srcnn_model-10-0.0252.pth")  # Load a model if available
    input_path = "./dataset/Test/00001.jpg"
    output_path = "./out.jpg"
    
    img = cv2.imread(input_path)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    # Enhance image
    enhanced_image = srcnn_wrapper.enhance_image(img)

    # Save or display the output
    srcnn_wrapper.save_image(enhanced_image, output_path)

