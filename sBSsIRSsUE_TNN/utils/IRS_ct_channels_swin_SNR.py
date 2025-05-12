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

    H_c = batch_khatri_rao(H_bi.permute(0,2,1), H_iu)
    h_mean = H_c.mean()
    h_std = H_c.std()

    if phase == 'train':
        H_c = (H_c - h_mean) / h_std
        print('Training data')
    h_c = vec(H_c)
    
    Psi = get_IRS_coef(IRScoef, n_R, n_I, n_T, T).to(device).to(torch.complex64) # IRS coefficient
    sgnl = vec(torch.matmul(H_c, Psi))  # Transmitted signal (vectorized)

    Ps = (torch.eye(n_T).to(device).abs()**2).sum()/n_T############################## Power of the transmitted signal
    if IRScoef in ['identity', 'i']:
        print(Ps)
    Pn = Ps / SNR_lin # Noise power
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2) # Repeat the noise power for each data sample groups (8)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device) # Generate noise
    y = sgnl + w # Received signal
    return turnReal(h_c), turnReal(y), h_mean, h_std # Return the channel, received signal, mean and std of the channel
