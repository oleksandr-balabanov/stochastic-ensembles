"""
	CREATE THE MODEL (PYRO)
"""

import torch
import torch.nn.functional as F

import pyro
from pyro.distributions import Normal
from pyro.nn import PyroModule, PyroSample
from pyro.nn.module import to_pyro_module_

import os

# import torch template
import stochastic_methods.model as model

# pyro posterior
class ToyNetPyroPosterior(model.ToyNet, PyroModule):
    def __init__(
        self, num_hidden_layers=1, hidden_dim=10, input_dim=2, num_classes=2, device="cpu"
    ):
        super(ToyNetPyroPosterior, self).__init__(
            num_hidden_layers, hidden_dim, input_dim, num_classes
        )

        # device
        self.device = device

        # module list to pyro
        to_pyro_module_(self.layers.to(self.device))

        # torch layers to pyro normal layers
        self.to_pyro_normal(self.input)
        [self.to_pyro_normal(layer) for layer in self.layers]
        self.to_pyro_normal(self.output)

    def to_device(self, device):
        self.device = device
        self.input.to(device)
        [layer.to(device) for layer in self.layers]
        self.output.to(device)

    def to_pyro_normal(self, layer):

        # convert torch layer to pyro
        to_pyro_module_(layer.to(self.device))
        mean = torch.tensor([0.0]).to(self.device)
        std = torch.tensor([1.0]).to(self.device)

        # parameters to distributions
        w_size = layer.weight.shape
        b_size = layer.bias.shape

        layer.weight = PyroSample(
            lambda self: Normal(mean, std).expand([w_size[0], w_size[1]]).to_event(2)
        )
        layer.bias = PyroSample(lambda self: Normal(mean, std).expand([b_size[0]]).to_event(1))

    def forward(self, x, y=None):

        batch = x.shape[0]

        x = F.relu(self.input(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        logits = self.output(x)

        # conditioned on the observed data
        with pyro.plate("data", batch):
            pyro.sample("obs", pyro.distributions.Categorical(logits=logits), obs=y)

        return logits
