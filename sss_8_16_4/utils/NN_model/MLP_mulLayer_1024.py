import torch.nn as nn

class MLP(nn.Module):
    """
    A simple multi-layers Multi-Layer Perceptron (MLP) neural network.
    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of neurons in the hidden layer.
        output_dim (int): The number of output features.
    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc...
        fco (nn.Linear): The output fully connected layer.
        activation (nn.Tanh): The activation function (Tanh) applied after each layer.
    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the network.
    """
    def __init__(self, input_dim, hidden_dim, ouput_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc9 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc10 = nn.Linear(hidden_dim, hidden_dim)
        self.fco = nn.Linear(hidden_dim, ouput_dim)
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        # x = self.activation(self.fc6(x))
        # x = self.activation(self.fc7(x))
        # x = self.activation(self.fc8(x))
        # x = self.activation(self.fc9(x))
        # x = self.activation(self.fc10(x))
        x = self.fco(x)
        return x 
    


# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         """
#         Args:
#         - input_dim (int): Input dimension.
#         - hidden_dim (int): Hidden layer dimension.
#         - output_dim (int): Output dimension.
#         """
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fco = nn.Linear(hidden_dim, output_dim)
#         self.activation = nn.Tanh()

#     def forward(self, x):
#         """
#         Forward pass with residual connection.
#         """
#         # First layer without residual connection
#         x = self.activation(self.fc1(x))

#         # Second layer with residual connection
#         residual = x  # Save the input for residual
#         x = self.activation(self.fc2(x))
#         x = x + residual  # Add residual connection

#         # Output layer
#         x = self.fco(x)
#         return x