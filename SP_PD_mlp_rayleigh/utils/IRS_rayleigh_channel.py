import torch
from utils.batch_khatri_rao import batch_khatri_rao
from utils.complex_utils import turnReal, turnCplx, vec
from utils.get_IRS_coef import get_IRS_coef
from utils.batch_kronecker import batch_kronecker

def importData(data_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='identity'):
    
    torch.manual_seed(data_size)

    W_Mean = 0
    sqrt2 = 2**0.5
    H_mean, H_std = [0,1]
    
    H_bi = torch.view_as_complex(torch.normal(H_mean, H_std, size=(data_size, n_I, n_T, 2))/sqrt2).to(device)
    H_iu = torch.view_as_complex(torch.normal(H_mean, H_std, size=(data_size, n_R, n_I, 2))/sqrt2).to(device)
    # H_iu = torch.ones(data_size, n_R, n_I, dtype=torch.complex64).to(device)
    H_c = batch_khatri_rao(H_bi.permute(0,2,1), H_iu)
    H_c = torch.view_as_complex(torch.normal(H_mean, H_std, size=(data_size, n_T*n_R, n_I, 2))/sqrt2).to(device)
    h_c = vec(H_c)
    # h_c = H_c.reshape(data_size, n_R*n_T*n_I)
    h_mean = h_c.mean()
    h_std = h_c.std()

    Psi = get_IRS_coef(IRScoef, n_R, n_I, n_T, T).to(device).to(torch.complex64)
    Sgnl = torch.matmul(H_c, Psi)
    sgnl = vec(Sgnl)
    # sgnl = Sgnl.reshape(data_size, n_R*T)
    sgnl_ = torch.matmul(batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64), h_c.unsqueeze(2)).squeeze(2)
    # print(torch.allclose(sgnl, sgnl_, atol=1e-6))

    Ps = (sgnl.abs()**2).mean()
    Pn = Ps / SNR_lin
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device)
    y = sgnl + w
    return turnReal(h_c), turnReal(y), h_mean, h_std