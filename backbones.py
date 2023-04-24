import torch
import torch.nn as nn 
from torchvision import models, datasets
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class HENN_EfficientNet(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super().__init__()
        self.pretrain = pretrain
        self.model_name = "efficientnet-b3"
        if self.pretrain:
            self.network = EfficientNet.from_pretrained(self.model_name, num_classes=num_classes)
        else:
            self.network = EfficientNet.from_name(self.model_name, num_classes=num_classes)
        self.output = nn.Softplus()

    def forward(self, x):
        logits = self.network(x)
        logits = self.output(logits)
        return logits


class EfficientNet_pretrain(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super().__init__()
        self.pretrain = pretrain
        self.model_name = "efficientnet-b3"
        if self.pretrain:
            self.network = EfficientNet.from_pretrained(self.model_name, num_classes=num_classes)
        else:
            self.network = EfficientNet.from_name(self.model_name, num_classes=num_classes)

    def forward(self, x):
        logits = self.network(x)
        return logits


class HENN_ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ResNet
        self.network = models.resnet50(pretrained=True)
        self.network.fc = torch.nn.Linear(2048, num_classes) 
        self.output = nn.Softplus()
        
    def forward(self, x):
        logits = self.network(x)
        logits = self.output(logits)
        return logits


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ResNet
        self.network = models.resnet50(pretrained=True)
        self.network.fc = torch.nn.Linear(2048, num_classes) 

    def forward(self, x):
        logits = self.network(x)
        return logits


class HENN_VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # VGG16
        self.network = models.vgg16(pretrained=True)
        self.network.classifier[6] = torch.nn.Linear(4096, num_classes)
        self.output = nn.Softplus()
        
    def forward(self, x):
        logits = self.network(x)
        logits = self.output(logits)
        return logits


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # VGG16
        self.network = models.vgg16(pretrained=True)
        self.network.classifier[6] = torch.nn.Linear(4096, num_classes)

    def forward(self, x):
        logits = self.network(x)
        # logits = self.output(logits)
        return logits


class HENN_LeNet(nn.Module):
    def __init__(self):
        super(HENN_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.output = nn.Softplus()
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = self.output(x)
        return x