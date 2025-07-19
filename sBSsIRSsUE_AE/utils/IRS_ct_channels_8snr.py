import torch
import scipy.io as scio
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_path, '..')))
from utils.complex_utils import turn_real, vec
from utils.batch_khatri_rao import batch_khatri_rao
from utils.get_IRS_coef import get_IRS_coef


def import_data(data_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='identity', phase = 'train', channel='default'):
    # torch.manual_seed(0)

    W_Mean = 0
    sqrt2 = 2**0.5
    current_path = os.path.dirname(os.path.abspath(__file__))

    ## generate communication data to train the parameterized policy
    if channel.lower() == 'uma':
        if phase == 'train':
            BI_file_name = 'UMa_BI_train_1M_4_8_.mat'
            IU_file_name = 'UMa_IU_train_1M_8_4_.mat'
        elif phase == 'val':
            BI_file_name = 'UMa_BI_val_2k_4_8_.mat'
            IU_file_name = 'UMa_IU_val_2k_8_4_.mat'
        elif phase == 'test':
            BI_file_name = 'UMa_BI_test_24k_4_8_.mat'
            IU_file_name = 'UMa_IU_test_24k_8_4_.mat'
    elif channel.lower() == 'inf':
        if phase == 'train':
            BI_file_name = 'InF_BI_train_1M_4_8_.mat'
            IU_file_name = 'InF_IU_train_1M_8_4_.mat'
        elif phase == 'val':
            BI_file_name = 'InF_BI_val_2k_4_8_.mat'
            IU_file_name = 'InF_IU_val_2k_8_4_.mat'
        elif phase == 'test':
            BI_file_name = 'InF_BI_test_24k_4_8_.mat'
            IU_file_name = 'InF_IU_test_24k_8_4_.mat'
    else:
        raise NameError(f"{channel} is not a valid channel name")
    
    dataset_folder = os.path.abspath(os.path.join(current_path, '..', '..', 'dataset_ct_realization'))
    for idx, file_name in enumerate([BI_file_name, IU_file_name]):
        if not os.path.exists(os.path.join(dataset_folder, file_name)):
            raise FileNotFoundError(f"File {file_name} not found in {dataset_folder}. Please check the dataset path.")
        dataset_file_path = os.path.join(dataset_folder, file_name)
        if   idx == 0: H_bi = torch.tensor(scio.loadmat(dataset_file_path)['H_samples']).to(torch.complex64).to(device)
        elif idx == 1: H_iu = torch.tensor(scio.loadmat(dataset_file_path)['H_samples']).to(torch.complex64).to(device)

    H_c = batch_khatri_rao(H_bi.permute(0,2,1), H_iu)#*10
    # use the same channel realization for all SNR groups
    # if phase == 'test':
    if phase == 'train':
        data_size_ = data_size // len(SNR_lin)
        H_c = H_c[:data_size_]
        H_c = H_c.repeat(len(SNR_lin), 1, 1) # Repeat the channel for each SNR group (8)
    elif phase == 'val':
        data_size_ = data_size // len(SNR_lin)
        H_c = H_c[:data_size_]
        H_c = H_c.repeat(len(SNR_lin), 1, 1) # Repeat the channel for each SNR group (8)
    elif phase == 'test':
        data_size_ = data_size // len(SNR_lin)
        H_c = H_c[:data_size_]
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

    Ps = (sgnl.abs()**2).mean() # Power of the transmitted signal
    Pn = Ps / SNR_lin # Noise power
    # print(f'Noise power: {Pn}')
    # if phase == 'val':
    #     data_size = data_size * len(SNR_lin)
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2) # Repeat the noise power for each data sample groups (8)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device) # Generate noise
    y = sgnl + w # Received signal
    return turn_real(h_c), turn_real(y), h_mean, h_std # Return the channel, received signal, mean and std of the channel
