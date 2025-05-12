import torch
import torch.nn as nn

class DynamicTanh(nn.Module):
    def __init__(self, num_features, alpha_init_value=1):
        super().__init__()
        # self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features)*1e-1)
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x = torch.tanh(self.alpha * x)
        x = torch.tanh(x)
        return x * self.weight + self.bias
        # return x
