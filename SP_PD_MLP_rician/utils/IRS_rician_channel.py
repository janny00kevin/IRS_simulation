import torch
import numpy as np
from utils.batch_khatri_rao import batch_khatri_rao
from utils.complex_utils import turnReal, turnCplx, vec
from utils.get_IRS_coef import get_IRS_coef
from utils.batch_kronecker import batch_kronecker

def importData(data_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='identity', case = 'train'):
    
    torch.manual_seed(data_size)

    W_Mean = 0
    sqrt2 = 2**0.5
    H_mean, H_std = [0,1]

    theta_B = torch.tensor(0.0)
    theta_I = torch.tensor(0.0)
    phi_I = torch.tensor(0.0)
    beta_BI = torch.tensor(1e6)
    d_bi = torch.tensor(51.0)
    d_iu = torch.tensor(3.0)
    C_0 = torch.tensor(0.001)
    alpha_bi = torch.tensor(2.0)
    alpha_iu = torch.tensor(2.8)

    # Total IRS elements
    n_Ix , n_Iz = 2, 4
    assert n_I == n_Ix * n_Iz, 'n_I is incorrect'
    # Array response vector for the BS
    a_B = torch.tensor(
        [np.exp(1j * np.pi * n * torch.sin(theta_B)) for n in range(n_T)], dtype=torch.cfloat
        ).reshape(-1, 1)  # Shape: (N_T, 1)
    # Array response vector for the IRS
    a_Ix = torch.tensor(
        [np.exp(1j * np.pi * n * torch.sin(theta_I) * torch.cos(phi_I)) for n in range(n_Ix)],dtype=torch.cfloat
        ).reshape(-1, 1)  # Shape: (N_Ix, 1)
    a_Iz = torch.tensor(
        [np.exp(1j * np.pi * n * torch.cos(theta_I) * torch.sin(phi_I)) for n in range(n_Iz)],dtype=torch.cfloat
        ).reshape(-1, 1)  # Shape: (N_Iz, 1)
    a_I = torch.kron(a_Ix, a_Iz)  # Shape: (N_I, 1)
    # LoS component
    G_LoS = torch.matmul(a_I, a_B.T)  # Shape: (N_I, N_T)
    # NLoS component (Rayleigh fading)
    G_NLoS = torch.view_as_complex(torch.randn(n_I, n_T, 2) / np.sqrt(2))  # i.i.d. Rayleigh fading
    # Combine LoS and NLoS components
    H_bi = (torch.sqrt(beta_BI / (1 + beta_BI)) * G_LoS + torch.sqrt(1 / (1 + beta_BI)) * G_NLoS).to(device)
    H_bi = (H_bi *torch.sqrt(C_0 * (d_bi) ** (-alpha_bi)))
    # print((d_bi) ** (-alpha_bi))

    # H_bi = torch.view_as_complex(torch.normal(H_mean, H_std, size=(data_size, n_I, n_T, 2))/sqrt2).to(device)
    H_iu = torch.view_as_complex(torch.normal(H_mean, H_std, size=(data_size, n_R, n_I, 2))/sqrt2).to(device)
    H_iu = H_iu* torch.sqrt(C_0 * (d_iu) ** (-alpha_iu)).to(device)
    # print('testing for no path loss!!!')
    # H_iu = torch.ones(data_size, n_R, n_I, dtype=torch.complex64).to(device)
    H_c = batch_khatri_rao(H_bi.unsqueeze(0).repeat(data_size, 1, 1).permute(0,2,1), H_iu)
    # H_c = torch.view_as_complex(torch.normal(H_mean, H_std, size=(data_size, n_T*n_R, n_I, 2))/sqrt2).to(device)
    h_mean = H_c.mean()
    h_std = H_c.std()
    ############################################################################
    if case == 'train':
        H_c = (H_c - h_mean) / h_std
        print('Training data')
    ############################################################################
    h_c = vec(H_c)

    Psi = get_IRS_coef(IRScoef, n_R, n_I, n_T, T).to(device).to(torch.complex64)
    Sgnl = torch.matmul(H_c, Psi)
    sgnl = vec(Sgnl)
    # print(sgnl.shape)
    # sgnl = Sgnl.reshape(data_size, n_R*T)
    sgnl_ = torch.matmul(batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64), h_c.unsqueeze(2)).squeeze(2)
    # print(torch.allclose(sgnl, sgnl_, atol=1e-6))

    Ps = (sgnl.abs()**2).mean()
    Pn = Ps / SNR_lin
    # print(Pn)
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device)
    y = sgnl + w
    return turnReal(h_c), turnReal(y), h_mean, h_std