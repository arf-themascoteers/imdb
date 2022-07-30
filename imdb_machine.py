import torch.nn as nn
import torch.nn.functional as F
import torchvision


class IMDBMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        number_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(number_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        for param in self.resnet.layer1.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.resnet(x)
        return x

