import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import cv2

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
        output_image = x.squeeze(0).cpu()
        output_image = ToPILImage()(output_image)
        output_image.save("./input.jpg")
        x = self.upsample(x)  # Upsample before feature extraction
        output_image = x.squeeze(0).cpu()
        output_image = ToPILImage()(output_image)
        output_image.save("./interm.jpg")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class SRCNNWrapper:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SRCNN().to(self.device)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def load_model(self, model_path):
        """Load pre-trained model weights."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")

    def preprocess_image(self, image):
        """Preprocess the image for SRCNN."""
        image = ToTensor()(image).unsqueeze(0).to(self.device)  # Add batch dimension
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
    srcnn_wrapper = SRCNNWrapper(model_path="./srcnn_model.pth")  # Load a model if available
    input_path = "./dataset/MineCraft-RT_1280x720_v14/MineCraft-RT_1280x720_v14/images/00000.jpg"
    output_path = "./out.jpg"
    
    img = cv2.imread(input_path)
    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Enhance image
    enhanced_image = srcnn_wrapper.enhance_image(img)

    # Save or display the output
    srcnn_wrapper.save_image(enhanced_image, output_path)

