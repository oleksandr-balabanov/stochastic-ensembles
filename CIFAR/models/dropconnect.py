"""

RESNET20-FRN with DropConnect modification

The base RESNET model is adapted from github.com/akamaster/pytorch_resnet_cifar10 by Yerlan Idelbayev

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utilities import DropConnectConv2d, DropConnectLinear, FRN, LambdaLayer, _weights_init

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', drop_rate = 0):
        super(BasicBlock, self).__init__()
        if stride == 1:
            padding = [1, 1, 1, 1]
        else:
            padding = [0, 1, 0, 1]
            
        self.pad1 = nn.ZeroPad2d(padding)
        self.conv1 = DropConnectConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=0, bias=True, drop_rate=drop_rate)
        self.bn1 = FRN(planes)
        self.conv2 =  DropConnectConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True, drop_rate=drop_rate)
        self.bn2 = FRN(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, s0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     DropConnectConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True, drop_rate=drop_rate),
                     FRN(self.expansion * planes)
                )

    def forward(self, x):
        out = self.pad1(x)
        out = F.silu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.silu(out)
        return out


class DropConnectResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, drop_rate_conv = 0, drop_rate_linear = 0):
        super(DropConnectResNet, self).__init__()
        self.drop_rate_conv = drop_rate_conv
        self.in_planes = 16

        self.conv1 = DropConnectConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True, drop_rate = drop_rate_conv)
        self.bn1 = FRN(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = DropConnectLinear(64, num_classes, drop_rate_linear)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, drop_rate = self.drop_rate_conv))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def dropconnect_resnet20(**kwargs):
    return DropConnectResNet(BasicBlock, [3, 3, 3], **kwargs)


def dropconnect_resnet32(**kwargs):
    return DropConnectResNet(BasicBlock, [5, 5, 5], **kwargs)


def dropconnect_resnet44(**kwargs):
    return DropConnectResNet(BasicBlock, [7, 7, 7], **kwargs)


def dropconnect_resnet56(**kwargs):
    return DropConnectResNet(BasicBlock, [9, 9, 9], **kwargs)


def dropconnect_resnet110(**kwargs):
    return DropConnectResNet(BasicBlock, [18, 18, 18], **kwargs)


def dropconnect_resnet1202(**kwargs):
    return DropConnectResNet(BasicBlock, [200, 200, 200], **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))
