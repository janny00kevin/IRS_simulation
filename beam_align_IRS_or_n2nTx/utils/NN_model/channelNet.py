import torch.nn as nn

class channelNet(nn.Module):
    def __init__(self, n_R=4, n_I=8, n_T=4, T=32, filt_size=3, num_channel=256, input_channels=3, ReLU=True):
        super(channelNet, self).__init__()

        self.is_relu = ReLU
        
        # Convolutional layers
        self.cnn1 = nn.Conv2d(input_channels, num_channel, kernel_size=filt_size, padding='same')
        # self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.cnn2 = nn.Conv2d(num_channel, num_channel, kernel_size=filt_size, padding='same')
        # self.bn2 = nn.BatchNorm2d(num_filters)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.cnn3 = nn.Conv2d(num_channel, num_channel, kernel_size=filt_size, padding='same')
        # self.bn3 = nn.BatchNorm2d(num_filters)
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(num_channel*n_R*T, 1024)  # Adjust based on flattened dimension
        self.dp = nn.Dropout(p=0.5)
        
        # Fully connected layer 2
        self.fc2 = nn.Linear(1024, 2048)
        # self.dp2 = nn.Dropout(p=0.5)

        # Final output layer
        self.fc3 = nn.Linear(2048, 2*n_R*n_I*n_T)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu(out) if self.is_relu else out
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu(out) if self.is_relu else out
        # Convolution 3
        out = self.cnn3(out)
        out = self.relu(out) if self.is_relu else out

        # Flatten for fully connected layers
        out = out.view(out.size(0), -1)
        
        # Fully connected layer 1
        out = self.fc1(out)
        out = self.relu(out) if self.is_relu else out
        out = self.dp(out)
        # Fully connected layer 2
        out = self.fc2(out)
        out = self.relu(out) if self.is_relu else out
        out = self.dp(out)

        # Output layer
        out = self.fc3(out)
        return out