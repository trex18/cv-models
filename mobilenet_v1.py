import torch
import torch.nn as nn
import torchvision
import os
from train import train
from transform import get_transform

# a layer consists of [input_channels, output_channels, depth, stride at 3x3 convolution]
DW_LAYERS_CONFIG = [
    [32, 64, 1, 1],
    [64, 128, 1, 2],
    [128, 128, 1, 1],
    [128, 256, 1, 2],
    [256, 256, 1, 1],
    [256, 512, 1, 2],
    [512, 512, 5, 1],
    [512, 1024, 1, 2],
    [1024, 1024, 1, 1] # looks like there was a typo in the paper, should be stride=1 instead of stride=2
]

class DepthwiseConv(nn.Module):
    def __init__(self, input_channels, output_channels, stride, padding=1):
        super(DepthwiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=stride, padding=1, groups=input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class MobileNet_v1(nn.Module):
    def __init__(self, dw_layers_config=DW_LAYERS_CONFIG):
        super(MobileNet_v1, self).__init__()
        self.dw_layers_config = dw_layers_config

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dw_layers = self.make_dw_layers()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 1000)

    def make_dw_layers(self):
        layers = []
        for layer in self.dw_layers_config:
            input_channels, output_channels, depth, stride = layer
            layers.append(DepthwiseConv(input_channels, output_channels, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dw_layers(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    DEVICE = torch.device('cuda')
    ROOT = 'data'
    NUM_EPOCHS = 10
    LR = 0.0001
    MOMENTUM = 0.9

    transforms = {phase: get_transform(phase, IMAGE_SIZE) for phase in ['train', 'val']}
    datasets = {
        phase: torchvision.datasets.ImageFolder(os.path.join(ROOT, phase), transform=transforms[phase])
        for phase in ['train', 'val']
    }

    dataloaders = {
        phase: torch.utils.data.DataLoader(datasets[phase], batch_size=BATCH_SIZE, shuffle=True) for phase in
        ['train', 'val']
    }
    model = MobileNet_v1(dw_layers_config=DW_LAYERS_CONFIG)
    model.fc = nn.Linear(1024, 10)
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model, accuracy_history, loss_history = train(model, NUM_EPOCHS, dataloaders, BATCH_SIZE, opt, criterion, DEVICE)