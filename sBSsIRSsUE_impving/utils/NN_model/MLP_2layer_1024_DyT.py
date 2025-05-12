import torch.nn as nn
from utils.NN_model.DynamicTanh import DynamicTanh

class MLP(nn.Module):
    """
    A simple 2-layer Multi-Layer Perceptron (MLP) neural network.
    
    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of neurons in the hidden layer.
        output_dim (int): Number of output features.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        dyt1 (DynamicTanh): Dynamic Tanh activation after the first layer.
        fc2 (nn.Linear): Second fully connected layer.
        dyt2 (DynamicTanh): Dynamic Tanh activation after the second layer.
        fco (nn.Linear): Output fully connected layer.
    
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass of the network.
    """
    def __init__(self, input_dim, hidden_dim, ouput_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dyt1 = DynamicTanh(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.dyt2 = DynamicTanh(hidden_dim)
        self.fco = nn.Linear(hidden_dim, ouput_dim)
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.dyt1(self.fc1(x))
        # x = self.dropout(x)
        x = self.activation(self.fc2(x))
        # x = self.dropout(x)
        x = self.fco(x)
        return x 
