import torch
import torch.nn as nn 
from torchvision import models, datasets
from efficientnet_pytorch import EfficientNet

class HENN_EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # EfficientNet
        self.network = EfficientNet.from_pretrained("efficientnet-b3", num_classes=num_classes)
        self.output = nn.Softplus()
        
    def forward(self, x):
        logits = self.network(x)
        logits = self.output(logits)
        return logits


class EfficientNet_pretrain(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # EfficientNet
        self.network = EfficientNet.from_pretrained("efficientnet-b3", num_classes=num_classes)

    def forward(self, x):
        logits = self.network(x)
        return logits


class HENN_ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # EfficientNet
        self.network = models.resnet50(pretrained=True)
        self.network.fc = torch.nn.Linear(2048, num_classes) # kappa 
        self.output = nn.Softplus()
        
    def forward(self, x):
        logits = self.network(x)
        logits = self.output(logits)
        return logits


class HENN_VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # EfficientNet
        self.network = models.vgg16(pretrained=True)
        self.network.classifier[6] = torch.nn.Linear(4096, num_classes)
        self.output = nn.Softplus()
        
    def forward(self, x):
        logits = self.network(x)
        logits = self.output(logits)
        return logits
