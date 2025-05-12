import torch
import scipy.io as scio
from utils.complex_utils import turnReal, vec
from utils.batch_khatri_rao import batch_khatri_rao
from utils.get_IRS_coef import get_IRS_coef
import time


def importData(data_size, device, phase = 'train', channel='default'):

    # W_Mean = 0
    # sqrt2 = 2**0.5

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
    assert int(data_size) == int(H_c.shape[0]), f"Data size %i does not match the channel data size %i!" % (data_size, H_c.shape[0])
    h_mean = H_c.mean()
    h_std = H_c.std()

    if phase == 'train':
        H_c = (H_c - h_mean) / h_std
        print('Training data')
    h_c = vec(H_c)

    return turnReal(h_c), h_mean, h_std # Return the channel, received signal, mean and std of the channel

def add_noise(recieve_sign, snr_lin, data_size, n_R, T, device, seed):
    torch.manual_seed(recieve_sign[0,0].real.int().item() + seed) # Set the seed for reproducibility
    # torch.manual_seed(recieve_sign[0,0].real.int().item())
    # if (seed-1)%1000 == 0:
    #     print('recieve_sign', recieve_sign[0,0].real.int().item() + seed)
    # Generate noise
    W_Mean = 0
    sqrt2 = 2**0.5

    Ps = (recieve_sign.abs()**2).mean() # Power of the transmitted signal
    Pn = Ps / snr_lin # Noise power
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(snr_lin)).reshape(data_size, n_R*T, 2) # Repeat the noise power for each data sample groups (8)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device) # Generate noise
    y = recieve_sign + w # Received signal
    return y
