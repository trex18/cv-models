import torch
import torch.nn as nn
import torchvision
import os
from train import train
from transform import get_transform
from math import ceil

b0_mbconv_config = [
    # expand_ratio, out_channels, kernel_size, stride, repeats
    [1, 16, 3, 1, 1],
    [6, 24, 3, 2, 2],
    [6, 40, 5, 2, 2],
    [6, 80, 3, 2, 3],
    [6, 112, 5, 1, 3],
    [6, 192, 5, 2, 4],
    [6, 320, 3, 1, 1]
]

scaled_models_parameters = {
    # phi, resolution, dropout rate (increases linearly from 0.2 to 0.5)
    # depth = alpha^phi, width = beta^phi
    'b0' : [0, 224, 0.2],
    'b1' : [0.5, 240, 0.243],
    'b2' : [1, 260, 0.285],
    'b3' : [2, 300, 0.328],
    'b4' : [3, 380, 0.371],
    'b5' : [4, 456, 0.414],
    'b6' : [5, 528, 0.457],
    'b7' : [6, 600, 0.5]
}

def make_config(model_name, b0_config=b0_mbconv_config, model_params=scaled_models_parameters, alpha=1.2, beta=1.1):
    """
    :param model_name: one of ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    alpha and beta are defined in the paper
    :return: config for one of the efficientnet models (of the same structure as b0_mbconv_config),
            resolution and dropout_rate
    """
    if model_name == 'b0':
        return b0_config, 224, 0.2
    phi, resolution, dropout_rate = model_params[model_name]
    layers = []
    for block in b0_config:
        expand_ratio, out_channels, kernel_size, stride, repeats = block
        out_channels = 4 * ceil(out_channels * beta**phi / 4)
        repeats = ceil(repeats * alpha**phi)
        layers.append([expand_ratio, out_channels, kernel_size, stride, repeats])
    return layers, resolution, dropout_rate

def conv_block(in_channels, out_channels, kernel_size, stride=1, groups=1, activation=nn.SiLU):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, groups=groups),
        nn.BatchNorm2d(out_channels),
        activation()
    )

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeezed_channels = in_channels // 4
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeezed_channels, 1),
            nn.SiLU(),
            nn.Conv2d(squeezed_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.scale(x)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, survival_prob=0.8):
        super(MBConv, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = stride == 1 and in_channels == out_channels
        expand_channels = in_channels * expand_ratio
        self.expand_1x1 = conv_block(in_channels, expand_channels, 1)
        self.dconv = conv_block(expand_channels, expand_channels, kernel_size, stride, groups=expand_channels)
        self.se = SqueezeExcitation(expand_channels)
        self.squeeze_1x1 = conv_block(expand_channels, out_channels, 1)

    def stochastic_depth(self, x, survival_prob):
        assert survival_prob > 0 and survival_prob <= 1
        if not self.training:
            return x
        mask = torch.bernoulli(survival_prob * torch.ones(x.shape[0], 1, 1, 1)).to(x.device)
        return torch.div(x, survival_prob) * mask

    def forward(self, x):
        residual = x
        x = self.expand_1x1(x)
        x = self.dconv(x)
        x = self.se(x)
        out = self.squeeze_1x1(x)
        if self.use_residual:
            out = self.stochastic_depth(out, self.survival_prob)
            out += residual
        return out

class EfficientNet(nn.Module):
    def __init__(self, model_name, num_classes):
        super(EfficientNet, self).__init__()
        self.model_name = model_name
        self.features, self.resolution, self.dropout_rate = self.make_features()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(1280, num_classes)

    def make_features(self):
        mbconv_config, resolution, dropout_rate = make_config(self.model_name)
        layers = []
        layers.append(conv_block(3, 32, 3, 2))
        in_channels = 32
        for block_params in mbconv_config:
            expand_ratio, out_channels, kernel_size, stride, repeats = block_params
            for i in range(repeats):
                layers.append(
                    MBConv(in_channels, out_channels, expand_ratio, kernel_size, stride)
                )
                in_channels = out_channels
                stride = 1   #only first MbConv in a series has stride > 1, the rest have stride=1
        layers.append(conv_block(in_channels, 1280, 1, 1))
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers), resolution, dropout_rate

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        out = self.fc(x)
        return out


if __name__ == "__main__":
    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    DEVICE = torch.device('cuda')
    ROOT = 'data'
    NUM_EPOCHS = 10
    LR = 0.001
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
    model = EfficientNet("b0", num_classes=10)
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model, accuracy_history, loss_history = train(model, NUM_EPOCHS, dataloaders, BATCH_SIZE, opt, criterion, DEVICE)