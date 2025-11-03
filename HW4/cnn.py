import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        """
        Instantiates the CNN model

        HINT: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        super(CNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Convolutional layers
            ...
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            # Linear layers
            ...
        )
        """
        super(CNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            # First layer
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            
            # Second layer
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            
            # Third layer
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            
            # Fourth layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2))
        
        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # First layer
            nn.Linear(64 * 7 * 7, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            
            # Second layer
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            
            # Final layer
            nn.Linear(128, 4))

    def forward(self, x):
        """
        Runs the forward method for the CNN model

        Args:
            x (torch.Tensor): input tensor to the model

        Returns:
            torch.Tensor: output classification tensor of the model
        """
        x = self.feature_extractor(x)
        x = self.avg_pooling(x)
        x = self.classifier(x)
        return x
