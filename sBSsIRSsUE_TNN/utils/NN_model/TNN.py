import torch
import torch.nn as nn
from utils.get_IRS_coef import get_IRS_coef

class TNN(nn.Module):
    def __init__(self, n_T, n_I, n_R):
        super(TNN, self).__init__()
        self.n_T = n_T
        self.n_R = n_R
        # self.n_I = n_I
        self.phase = nn.Parameter(torch.randn(n_I, n_I))

    def forward(self, h):
        # calculate (\Psi^T ⊗ \tbX) \h
        device = h.device  
        Psi_T = torch.exp(1j * self.phase).T.contiguous().to(device)  
        # Psi_T = get_IRS_coef('h', self.n_R, self.n_I, self.n_T, self.n_I*self.n_T).T.contiguous().to(torch.complex64).to(device)
        I_NtNr = torch.eye(self.n_T * self.n_R).to(device)  
        
        kron_product = torch.kron(Psi_T, I_NtNr)
        
        output = torch.matmul(kron_product.unsqueeze(0), h.unsqueeze(-1)).squeeze()
        return output