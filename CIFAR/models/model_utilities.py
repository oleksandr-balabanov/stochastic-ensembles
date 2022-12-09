"""

	UTILITIES 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os


# DropOutLinear layer
class DropOutLinear(nn.Module):
    def __init__(self, dim_in, dim_out, drop_rate):
        super(DropOutLinear, self).__init__()

        self.drop_rate = drop_rate
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):

        return F.dropout(self.linear(x), p=self.drop_rate, training=self.training)


# DropOutConv2d layer
class DropOutConv2d(nn.Module):
    def __init__(
        self, in_planes, planes, kernel_size=3, stride=1, padding=0, bias=True, drop_rate=0
    ):
        super(DropOutConv2d, self).__init__()

        self.drop_rate = drop_rate
        self.conv = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):

        return F.dropout(self.conv(x), p=self.drop_rate, training=self.training)


# NPDropOutLinear layer
class NPDropOutLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NPDropOutLinear, self).__init__()

        self.linear = nn.ModuleList()
        for swap in range(2):
            self.linear.append(nn.Linear(dim_in, dim_out))

    def mask(self, x):
        mask_int = torch.randint(2, (x[0].shape[1],))
        out = 0
        for swap in range(2):
            mask = (mask_int == swap).float().to(x[0].get_device())
            out += x[swap] * mask
        return out

    def forward(self, x):
        res = []
        for swap in range(2):
            res.append(self.linear[swap](x))
        return self.mask(res)


# NPDropOutConv2d layer
class NPDropOutConv2d(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=0, bias=True):
        super(NPDropOutConv2d, self).__init__()

        self.conv = nn.ModuleList()
        for swap in range(2):
            self.conv.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )

    def mask(self, x):
        mask_int = torch.randint(2, (x[0].shape[1], x[0].shape[2], x[0].shape[3]))

        out = 0
        for swap in range(2):
            mask = (mask_int == swap).float().to(x[0].get_device())
            out += x[swap] * mask
        return out

    def forward(self, x):
        res = []
        for swap in range(2):
            res.append(self.conv[swap](x))

        return self.mask(res)


# DropConnectLinear layer
class DropConnectLinear(nn.Module):
    def __init__(self, dim_in, dim_out, drop_rate):
        super(DropConnectLinear, self).__init__()

        self.drop_rate = drop_rate
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):

        # dropconnect weight
        w = torch.nn.functional.dropout(
            self.linear.weight, p=self.drop_rate, training=self.training
        )

        # gaussian bias
        b = torch.nn.functional.dropout(self.linear.bias, p=self.drop_rate, training=self.training)

        return F.linear(x, w, b)


# DropConnectConv2d layer
class DropConnectConv2d(nn.Module):
    def __init__(
        self, in_planes, planes, kernel_size=3, stride=1, padding=0, bias=True, drop_rate=0
    ):
        super(DropConnectConv2d, self).__init__()

        self.drop_rate = drop_rate
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):

        # dropconnect weight
        w = torch.nn.functional.dropout(self.conv.weight, p=self.drop_rate, training=self.training)

        # gaussian bias
        b = None
        if self.bias == True:
            b = torch.nn.functional.dropout(
                self.conv.bias, p=self.drop_rate, training=self.training
            )

        return F.conv2d(x, w, b, self.stride, self.padding)


# deterministic FRN layer
class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        The shape of weight, bias, tau is  [1, C, 1, 1].
        eps is a scalar constant or learnable parameter.

        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps

        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True
        )
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)

        self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.tau)

    def forward(self, x):

        # compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # weight and bias
        x = self.weight * x + self.bias
        return torch.max(x, self.tau)


# NPDropOutFRN layer
class NPDropOutFRN(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(NPDropOutFRN, self).__init__()

        self.frn = nn.ModuleList()
        for swap in range(2):
            self.frn.append(FRN(num_features=num_features, eps=eps))

    def mask(self, x):
        mask_int = torch.randint(2, (x[0].shape[1], x[0].shape[2], x[0].shape[3]))

        out = 0
        for swap in range(2):
            mask = (mask_int == swap).float().to(x[0].get_device())
            out += x[swap] * mask
        return out

    def forward(self, x):
        res = []
        for swap in range(2):
            res.append(self.frn[swap](x))

        return self.mask(res)


# lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# weight initialization
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
