import torch
import torch.nn as nn

from adaptive_activation import AdaAct


# Basic Block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        if type(activation) == list:
            for act in activation:
                assert isinstance(act, nn.Module)
            self.activation1 = AdaAct(activation)
            self.activation2 = AdaAct(activation)
        else:
            self.activation1 = activation
            self.activation2 = activation

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.activation2(out)

        return out

# ResNet-18 Model
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, activation=nn.ReLU()):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        if type(activation) == list:
            for act in activation:
                assert isinstance(act, nn.Module)
            self.activation1 = AdaAct(activation)
        else:
            self.activation1 = activation

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], activation=activation)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation=activation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, activation=nn.ReLU()):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, activation=activation))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Define ResNet-18
def ResNet18(num_classes=100, activation=nn.ReLU()):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, activation=activation)

