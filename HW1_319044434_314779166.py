import torch
import torch.nn as nn

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(OptimizedCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Reduced filters
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                  # Output: (16, 64, 64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Reduced filters
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                  # Output: (32, 32, 32)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduced filters
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                  # Output: (64, 16, 16)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Reduced filters
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                  # Output: (64, 8, 8)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Final convolution layer
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)                      # Global average pooling
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),                                 # Flatten the feature map
            nn.Linear(128, 64),                          # Input adjusted to Layer5 output
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)                   # Output layer for 4 classes
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc(out)
        return out
