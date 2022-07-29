import torch.nn as nn
import torch.nn.functional as F


class TocMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.machine = nn.Sequential(
            nn.Conv2d(3, 16, (16,16)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, (8, 8)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, (4, 4)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(23104, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
        )


    def forward(self, x):
        x = self.machine(x)
        return x

