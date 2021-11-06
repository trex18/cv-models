import torch
import torch.nn as nn
import torchvision
import os
from train import train
from transform import get_transform

# each layer consists of [expansion_ratio, output_channels, repeat_number, stride_of_first_sublayer]
bottleneck_layers_config = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1]
]

def conv_block(input_channels, output_channels, kernel_size, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups),
        nn.BatchNorm2d(output_channels),
        nn.ReLU6()
    )

class BottleneckResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, s, t):
        """
        :param s: int, stride
        :param t: int, expansion ratio
        """
        super(BottleneckResidualBlock, self).__init__()
        assert s in (1, 2)
        self.skip_connection = True if (s == 1 and input_channels == output_channels) else False
        inter_channels = t * input_channels
        self.conv1x1_expand = conv_block(input_channels, inter_channels, 1)
        self.conv3x3 = conv_block(inter_channels, inter_channels, 3, s, 1, inter_channels)
        self.conv1x1_squeeze = conv_block(inter_channels, output_channels, 1)


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        identity = x
        out = self.conv1x1_expand(x)
        out = self.conv3x3(out)
        out = self.conv1x1_squeeze(out)
        if self.skip_connection:
            out += identity
        return out

class MobileNet_v2(nn.Module):
    def __init__(self, bottleneck_config, num_classes):
        super(MobileNet_v2, self).__init__()
        self.conv1 = conv_block(3, 32, 3, 2, 1)
        self.bottleneck_layers = self._make_bottleneck_layers(bottleneck_config)
        self.conv_9 = conv_block(320, 1280, 1)
        self.avg_pool = nn.AvgPool2d(7)
        self.final_conv = nn.Conv2d(1280, num_classes, 1)
        self.flatten = nn.Flatten()

    def _make_bottleneck_layers(self, config):
        layers = []
        input_channels = 32
        for layer in config:
            t, c, n, s = layer
            layers.append(BottleneckResidualBlock(input_channels, c, s, t))
            for i in range(n - 1):
                layers.append(BottleneckResidualBlock(c, c, 1, t))
            input_channels = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottleneck_layers(x)
        x = self.conv_9(x)
        x = self.avg_pool(x)
        x = self.final_conv(x)
        x = self.flatten(x)
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
    model = MobileNet_v2(bottleneck_layers_config, num_classes=10)
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model, accuracy_history, loss_history = train(model, NUM_EPOCHS, dataloaders, BATCH_SIZE, opt, criterion, DEVICE)