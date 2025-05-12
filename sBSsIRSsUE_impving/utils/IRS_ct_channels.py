import torch
import scipy.io as scio
from utils.complex_utils import turnReal, vec
from utils.batch_khatri_rao import batch_khatri_rao
from utils.get_IRS_coef import get_IRS_coef
import time


def importData(data_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='identity', phase = 'train', channel='default'):

    W_Mean = 0
    sqrt2 = 2**0.5

    ## generate communication data to train the parameterized policy
    if channel.lower() == 'uma':
        if phase == 'train':
            BI_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/UMa_BI_train_1M_4_8_.mat'
            IU_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/UMa_IU_train_1M_8_4_.mat'
        elif phase == 'val':
            BI_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/UMa_BI_val_2k_4_8_.mat'
            IU_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/UMa_IU_val_2k_8_4_.mat'
        elif phase == 'test':
            BI_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/UMa_BI_test_24k_4_8_.mat'
            IU_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/UMa_IU_test_24k_8_4_.mat'
    elif channel.lower() == 'inf':
        if phase == 'train':
            BI_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/InF_BI_train_1M_4_8_.mat'
            IU_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/InF_IU_train_1M_8_4_.mat'
        elif phase == 'val':
            BI_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/InF_BI_val_2k_4_8_.mat'
            IU_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/InF_IU_val_2k_8_4_.mat'
        elif phase == 'test':
            BI_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/InF_BI_test_24k_4_8_.mat'
            IU_file_path = './IRS_simulation/sBSsIRSsUE_ct/channel/InF_IU_test_24k_8_4_.mat'
    else:
        raise NameError(f"{channel} is not a valid channel name")

    H_bi = torch.tensor(scio.loadmat(BI_file_path)['H_samples']).to(torch.complex64).to(device)
    # print(H_bi.shape)
    H_iu = torch.tensor(scio.loadmat(IU_file_path)['H_samples']).to(torch.complex64).to(device)
    # print(H_iu.shape)

    H_c = batch_khatri_rao(H_bi.permute(0,2,1), H_iu)#*10
    if phase == 'test':
        test_size = 3000
        H_c = H_c[:test_size]
        H_c = H_c.repeat(len(SNR_lin), 1, 1) # Repeat the channel for each SNR group (8)
    h_mean = H_c.mean()
    h_std = H_c.std()

    if phase == 'train':
        H_c = (H_c - h_mean) / h_std
        print('Training data')
    h_c = vec(H_c)
    
    if IRScoef in ['i', 'd', 'h']:
        Psi = get_IRS_coef(IRScoef, n_R, n_I, n_T, T).to(device).to(torch.complex64) # IRS coefficient
    elif IRScoef.shape[0] == n_I:
        Psi = IRScoef.to(device).to(torch.complex64)
    else:
        raise NameError(f"{IRScoef} is not a valid IRS coefficient name")
    # print(f'IRS coefficient: {Psi}')
    sgnl = vec(torch.matmul(H_c, Psi))  # Transmitted signal (vectorized)

    # Psi_T = Psi.T.contiguous().to(device)  
    # # Psi_T = get_IRS_coef('h', self.n_R, self.n_I, self.n_T, self.n_I*self.n_T).T.contiguous().to(torch.complex64).to(device)
    # I_NtNr = torch.eye(n_T * n_R).to(device)  
    # kron_product = torch.kron(Psi_T, I_NtNr)
    # sgnl = torch.matmul(kron_product.unsqueeze(0), h_c.unsqueeze(-1)).squeeze()

    Ps = (sgnl.abs()**2).mean() # Power of the transmitted signal
    # print(f'Power of the transmitted signal: {Ps}')
    Pn = Ps / SNR_lin # Noise power
    # print(f'Noise power: {Pn}')
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2) # Repeat the noise power for each data sample groups (8)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device) # Generate noise
    y = sgnl + w # Received signal
    return turnReal(h_c), turnReal(y), h_mean, h_std # Return the channel, received signal, mean and std of the channel
