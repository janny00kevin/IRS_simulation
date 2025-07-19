import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        super(SparseAutoencoder, self).__init__()
        if output_dim is None: output_dim = input_dim
        # self.hidden_dim = 1024
        self.enfc1 = nn.Linear(input_dim, hidden_dim)
        self.enfc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enfco = nn.Linear(hidden_dim, input_dim)

        self.defc1 = nn.Linear(input_dim, hidden_dim)
        self.defc2 = nn.Linear(hidden_dim, hidden_dim)
        self.defco = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.lamb = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, x):
        # Encoder
        latent = self.enfco(self.tanh(self.enfc2(self.tanh(self.enfc1(x)))))
        latent_sparse = soft_thres(latent, torch.abs(self.lamb))
        # latent_sparse = self.relu(latent)
        # Decoder
        reconstructed = self.defco(self.tanh(self.defc2(self.tanh(self.defc1(latent_sparse)))))
        return reconstructed, latent_sparse
    
class SparseAutoencoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        super(SparseAutoencoder2, self).__init__()
        if output_dim is None: output_dim = input_dim
        # self.hidden_dim = 1024
        self.enfc1 = nn.Linear(input_dim, hidden_dim)
        self.enfc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enfc3 = nn.Linear(hidden_dim, hidden_dim)
        self.enfc4 = nn.Linear(hidden_dim, hidden_dim)
        self.enfco = nn.Linear(hidden_dim, input_dim)

        self.defc1 = nn.Linear(input_dim, hidden_dim)
        self.defc2 = nn.Linear(hidden_dim, hidden_dim)
        self.defco = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.lamb = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, x):
        # Encoder
        # latent = self.enfco(self.tanh(self.enfc2(self.tanh(self.enfc1(x)))))
        latent = self.enfco(self.tanh(self.enfc4(self.tanh(self.enfc3(self.tanh(self.enfc2(self.tanh(self.enfc1(x)))))))))
        latent_sparse = soft_thres(latent, torch.abs(self.lamb))
        # latent_sparse = self.relu(latent)
        # Decoder
        reconstructed = self.defco(self.tanh(self.defc2(self.tanh(self.defc1(latent_sparse)))))
        return reconstructed, latent_sparse
    
class SparseAutoencoder0(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        super(SparseAutoencoder0, self).__init__()
        if output_dim is None: output_dim = input_dim
        # self.hidden_dim = 1024
        self.enfc1 = nn.Linear(input_dim, hidden_dim)
        self.enfco = nn.Linear(hidden_dim, input_dim)

        self.defc1 = nn.Linear(input_dim, hidden_dim)
        self.defco = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.lamb = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, x):
        # Encoder
        latent = self.enfco(self.tanh(self.enfc1(x)))
        latent_sparse = soft_thres(latent, torch.abs(self.lamb))
        # latent_sparse = self.relu(latent)
        # Decoder
        reconstructed = self.defco(self.tanh(self.defc1(latent_sparse)))
        return reconstructed, latent_sparse

def soft_thres(x, lamb):
    len_x = x.shape[1] // 2
    x_ = torch.complex(x[:,:len_x].T, x[:,len_x:].T)
    abs_x = torch.abs(x_)
    x_sparse_ = x_ * torch.relu(abs_x - lamb) / abs_x
    x_sparse_.real[torch.isnan(x_sparse_.real)] = 0
    x_sparse_.imag[torch.isnan(x_sparse_.imag)] = 0
    x_sparse = torch.concat((x_sparse_.real, x_sparse_.imag), dim=0).T
    return x_sparse