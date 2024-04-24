import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
import torch

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Apply He initialization (fan_out variant)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Load a pre-trained ResNet50 model
        self.ldim = 1024
        self.encoder = models.resnet18(pretrained=True)
        
        # Replace the last layer with a trainable layer for encoding
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, self.ldim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.ldim, 128, kernel_size=3, stride=3, padding=0, output_padding=0),  
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=0),  
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  

        )
        self.decoder.apply(initialize_weights)

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.encoder.fc(x)
        x = x.view(x.size(0), self.ldim, 1, 1)  # Reshape for the decoder

        x = self.decoder(x)
        return x
