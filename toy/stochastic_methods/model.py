"""
	CREATE MODEL (TORCH)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stochastic_methods.utilities.model_utilities import (
    NPDropOutLinear,
    DropConnectLinear,
    DropOutLinear,
)

import os


# choose model
def train_create_net(args):

    if (
            args.method == "regular" or args.method == "multiswa" or
            args.method == "multiswag"
        ):
        return ToyNetDropOut(
            num_hidden_layers=args.num_hidden_layers,
            hidden_dim=args.hidden_dim,
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            drop_rate=0,
        )

    if args.method == "dropout":
        return ToyNetDropOut(
            num_hidden_layers=args.num_hidden_layers,
            hidden_dim=args.hidden_dim,
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            drop_rate=args.drop_rate,
        )

    if args.method == "np_dropout":
        return ToyNetNPDropOut(
            num_hidden_layers=args.num_hidden_layers,
            hidden_dim=args.hidden_dim,
            input_dim=args.input_dim,
            num_classes=args.num_classes,
        )

    if args.method == "dropconnect":
        return ToyNetDropConnect(
            num_hidden_layers=args.num_hidden_layers,
            hidden_dim=args.hidden_dim,
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            drop_rate=args.drop_rate,
        )


# torch model
class ToyNet(nn.Module):
    def __init__(self, num_hidden_layers=1, hidden_dim=10, input_dim=2, num_classes=2):
        super(ToyNet, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
        )
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = F.relu(self.input(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        logits = self.output(x)

        return logits


# torch binary dropout model
class ToyNetDropOut(nn.Module):
    def __init__(
        self, num_hidden_layers=1, hidden_dim=10, input_dim=2, num_classes=2, drop_rate=0.1
    ):
        super(ToyNetDropOut, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.drop_rate = drop_rate

        #self.input = DropOutLinear(input_dim, hidden_dim, self.drop_rate)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                DropOutLinear(hidden_dim, hidden_dim, self.drop_rate)
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = F.relu(self.input(x))
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
        logits = self.output(x)

        return logits


# torch nonparametric dropout model
class ToyNetNPDropOut(nn.Module):
    def __init__(self, num_hidden_layers=1, hidden_dim=10, input_dim=2, num_classes=2):
        super(ToyNetNPDropOut, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        self.input = NPDropOutLinear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [NPDropOutLinear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
        )
        self.output = NPDropOutLinear(hidden_dim, num_classes)

    def forward(self, x):

        x = F.relu(self.input(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        logits = self.output(x)

        return logits


# torch drop connect model
class ToyNetDropConnect(nn.Module):
    def __init__(self, num_hidden_layers=1, hidden_dim=10, input_dim=2, num_classes=2, drop_rate=0):
        super(ToyNetDropConnect, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.drop_rate = drop_rate

        #self.input = DropConnectLinear(input_dim, hidden_dim, drop_rate)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                DropConnectLinear(hidden_dim, hidden_dim, drop_rate)
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = F.relu(self.input(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        logits = self.output(x)

        return logits
