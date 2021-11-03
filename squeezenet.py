import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from train import train
from transform import get_transform

layers_configuration = [
    [2, 16, 64, 64, 'complex'],
    [2, 16, 64, 64, 'simple'],
    [2, 32, 128, 128, 'complex'],
    'm',
    [2, 32, 128, 128, 'simple'],
    [2, 48, 192, 192, 'complex'],
    [2, 48, 192, 192, 'simple'],
    [2, 64, 256, 256, 'complex'],
    'm',
    [2, 64, 256, 256, 'simple']
]

class Fire(nn.Module):
    def __init__(self, input_channels, s1, e1, e3, skip_connection_type = None):
        """
        :param input_channels: int
        :param s1: int, squeeze layers
        :param e1: int, expand layers 1x1
        :param e3: int, expand layers 3x3
        :param skip_connection_type: str, either None, or 'simple', or 'complex'
        """
        super(Fire, self).__init__()
        self.squeeze_conv = nn.Conv2d(in_channels=input_channels, out_channels=s1, kernel_size=1)
        self.bn_squeeze = nn.BatchNorm2d(s1)
        self.expand_1x1 = nn.Conv2d(in_channels=s1, out_channels=e1, kernel_size=1)
        self.bn_1x1 = nn.BatchNorm2d(e1)
        self.expand_3x3 = nn.Conv2d(in_channels=s1, out_channels=e3, kernel_size=3, padding=1)
        self.bn_3x3 = nn.BatchNorm2d(e3)
        self.skip_connection_type = skip_connection_type
        if self.skip_connection_type == 'complex':
            self.skip_connection = nn.Conv2d(input_channels, e1+e3, kernel_size=1)
        elif self.skip_connection_type == 'simple':
            self.skip_connection = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.bn_squeeze(self.squeeze_conv(x))
        out_branch_1 = self.bn_1x1(self.expand_1x1(out))
        out_branch_2 = self.bn_3x3(self.expand_3x3(out))
        out = torch.cat((out_branch_1, out_branch_2), dim=1)
        out = F.relu(out)
        if self.skip_connection_type:
            out += self.skip_connection(identity)
        return out

class SqueezeNet(nn.Module):
    def __init__(self, configuration=layers_configuration, skip_connection_type=None):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        input_channels, self.fire_layers = self.create_fire_layers(configuration, skip_connection_type)
        self.dropout = nn.Dropout2d(0.5)
        self.conv10 = nn.Conv2d(input_channels, out_channels=1000, kernel_size=1, stride=1)
        self.avg_pool10 = nn.AvgPool2d(kernel_size=13, stride=1)
        self._init_weights()

    def create_fire_layers(self, configuration, network_skip_connection_type):
        layers = []
        input_channels = 96
        for layer in configuration:
            assert layer == 'm' or isinstance(layer, list)
            if layer == 'm':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            else:
                _, s1, e1, e3, skip_connection_type = layer
                skip_connection_type = self._skip_connection_type(network_skip_connection_type, skip_connection_type)
                layers.append(Fire(input_channels, s1, e1, e3, skip_connection_type))
                input_channels = e1 + e3
        return input_channels, nn.Sequential(*layers)

    def _skip_connection_type(self, network_skip_connection_type, skip_connection_type):
        """
        Every Fire block has some a type of skip connection predefined in configuration
        which is either simple or complex (i.e. identity or 1x1 conv).
        Skip connections are not used in the first type of Shufflenets, without skip connections.
        In the second type ("simple" ShuffleNets) only simple skip connections are used (identity).
        "complex" ShuffleNets use both simple and complex skip-connections.
        e.g. this function returns None for a "complex" skip-connection in a "simple" net
        :param network_skip_connection_type: str or None
        :param skip_connection_type: str or None
        :return: str or None
        """
        assert network_skip_connection_type in [None, 'simple', 'complex']
        assert skip_connection_type in ['simple', 'complex']
        if network_skip_connection_type is None:
            return None
        if skip_connection_type == 'simple':
            return 'simple'
        if network_skip_connection_type == 'complex':
            return 'complex'
        return None

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mp1(x)
        x = self.fire_layers(x)
        x = self.dropout(x)
        x = self.conv10(x)
        x = self.avg_pool10(x)
        return torch.squeeze(x)

if __name__ == "__main__":
    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    DEVICE = torch.device('cuda')
    ROOT = 'data'
    NUM_EPOCHS = 2
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
    model = SqueezeNet(skip_connection_type='complex')
    model.conv10 = nn.Conv2d(512, out_channels=10, kernel_size=1, stride=1)
    model.avg_pool10 = nn.AvgPool2d(kernel_size=13, stride=1)

    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model, accuracy_history, loss_history = train(model, NUM_EPOCHS, dataloaders, BATCH_SIZE, opt, criterion, DEVICE)