"""

	UTILITIES (TORCH)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# DropOut Linear layer
class DropOutLinear(nn.Module):
    def __init__(self, dim_in, dim_out, drop_rate):
        super(DropOutLinear, self).__init__()

        self.drop_rate = drop_rate
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):

        return F.dropout(self.linear(x), p=self.drop_rate, training=self.training)


# NPDropOut Linear layer
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


# DropConnect Linear layer
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
