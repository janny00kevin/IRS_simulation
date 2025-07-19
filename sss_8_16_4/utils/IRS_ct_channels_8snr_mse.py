import torch
# import scipy.io as scio
import h5py
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_path, '..')))
from utils.complex_utils import turn_real, vec
from utils.batch_khatri_rao import batch_khatri_rao, batch_khatri_rao_chunked
from utils.get_IRS_coef import get_IRS_coef


def import_data(data_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='identity', phase = 'train', channel='default', nmlz=False):

    W_Mean = 0
    sqrt2 = 2**0.5
    current_path = os.path.dirname(os.path.abspath(__file__))

    if phase == 'train':
        phase_ = 'train_1M'
    elif phase == 'val':
        phase_ = 'val_2k'
    elif phase == 'test':
        phase_ = 'test_24k'
    else:
        raise NameError(f"{channel} is not a valid channel name")
    BI_file_name = channel.lower() + '_BI_' + phase_ + '_8_16_ori_.mat'
    IU_file_name = channel.lower() + '_IU_' + phase_ + '_16_4_ori_.mat'
    
    dataset_folder = os.path.abspath(os.path.join(current_path, '..', '..', 'channel_realization_8_16_4'))
    for idx, file_name in enumerate([BI_file_name, IU_file_name]):
        if not os.path.exists(os.path.join(dataset_folder, file_name)):
            raise FileNotFoundError(f"File {file_name} not found in {dataset_folder}. Please check the dataset path.")
        dataset_file_path = os.path.join(dataset_folder, file_name)
        with h5py.File(dataset_file_path, 'r') as f:
            # h = f['H_samples']
            h = torch.complex(torch.tensor(f['H_samples']['real']), torch.tensor(f['H_samples']['imag']))
            # h = torch.tensor(f['H_samples'][:])
        # if   idx == 0: H_bi = torch.tensor(scio.loadmat(dataset_file_path)['H_samples']).to(torch.complex64).to(device)
        if   idx == 0: 
            H_bi = h.to(torch.complex64).permute(2,1,0)
            # print(f"BI channel shape: {H_bi.shape}")
        elif idx == 1: 
            H_iu = h.to(torch.complex64).permute(2,1,0)
            # print(f"IU channel shape: {H_iu.shape}")
    
    # del h
    # torch.cuda.empty_cache()

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

    if nmlz == True:
        H_c = (H_c - h_mean) / h_std
        print('Data normalized')
    elif nmlz == 'default':
        raise ValueError("Normalization factor nmlz must be 1 or 0, got {}".format(nmlz))
    h_c = vec(H_c)
    
    if IRScoef in ['i', 'd', 'h']:
        Psi = get_IRS_coef(IRScoef, n_R, n_I, n_T, T).to(torch.complex64) # IRS coefficient
    elif IRScoef.shape[0] == n_I:
        Psi = IRScoef.to(torch.complex64)
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
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn).to('cpu'))/sqrt2) # Generate noise
    y = sgnl + w # Received signal

    if phase != 'train':
        h_c = h_c.to(device)
        y = y.to(device)
    h_mean = h_mean.to(device)
    h_std = h_std.to(device)

    # Return the channel, received signal, mean and std of the channel
    return turn_real(h_c), turn_real(y), h_mean, h_std
