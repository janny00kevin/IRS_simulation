import torch
import scipy.io as scio
from utils.complex_utils import turnReal, vec
from utils.batch_khatri_rao import batch_khatri_rao
# from utils.get_IRS_coef import get_IRS_coef
from utils.steering_vector import steering_vector

def importData(data_size, n_R, n_T_x, n_T_y, T, SNR_lin, device, phase = 'train', channel='x', steering='x'):
    """
    Imports and processes channel data for IRS simulation.

    Args:
        data_size: Number of data samples.
        n_R: Number of receive antennas.
        n_I: Number of IRS elements.
        n_T: Number of transmit antennas.
        T: Time slots.
        SNR_lin: Linear SNR values.
        device: Torch device (CPU or GPU).
        IRScoef: IRS coefficient type ('x').
        phase: Data phase ('train', 'val', 'test').
        channel: Channel type ('uma').
        config: Configuration type ('original', 'aligned_23.1').

    Returns:
        A tuple containing:
            - Real part of the vectorized channel (h_c).
            - Real part of the received signal (y).
            - Mean of the channel (h_mean).
            - Standard deviation of the channel (h_std).

    Raises:
        ValueError: If invalid channel, config, or phase is provided.
    """
    torch.manual_seed(data_size)

    W_Mean = 0
    sqrt2 = 2**0.5
    n_T = n_T_x * n_T_y

    ## generate communication data to train the parameterized policy
    if channel.lower() == 'uma':
        # if config.lower() in ['original', 'o']:
        #     if phase == 'train':
        #         BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/UMa_BI_train_1M_4_8_ori_.mat'
        #         IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/UMa_IU_train_1M_8_1_ori_.mat'
        #     elif phase == 'val':
        #         BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/UMa_BI_val_2k_4_8_ori_.mat'
        #         IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/UMa_IU_val_2k_8_1_ori_.mat'
        #     elif phase == 'test':
        #         BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/UMa_BI_test_24k_4_8_ori_.mat'
        #         IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/UMa_IU_test_24k_8_1_ori_.mat'
        # if config.lower() in ['aligned_23.1', 'a']:
            if phase == 'train':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_BI_train_1M_4_8_23p1_.mat'
                # IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_IU_train_1M_8_1_23p1_.mat'
            elif phase == 'val':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_BI_val_2k_4_8_23p1_.mat'
                # IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_IU_val_2k_8_1_23p1_.mat'
            elif phase == 'test':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_BI_test_24k_4_8_23p1_.mat'
                # IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_IU_test_24k_8_1_23p1_.mat'
        # else:
        #     raise ValueError(f"{config} is not a valid configuration name")
    elif channel.lower() == 'inf':
        # if config.lower() in ['original', 'o']:
        #     if phase == 'train':
        #         BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/InF_BI_train_1M_4_8_ori_.mat'
        #         IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/InF_IU_train_1M_8_1_ori_.mat'
        #     elif phase == 'val':
        #         BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/InF_BI_val_2k_4_8_ori_.mat'
        #         IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/InF_IU_val_2k_8_1_ori_.mat'
        #     elif phase == 'test':
        #         BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/InF_BI_test_24k_4_8_ori_.mat'
        #         IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_ori/InF_IU_test_24k_8_1_ori_.mat'
        # elif config.lower() in ['aligned_23.1', 'a']:
            if phase == 'train':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_BI_train_1M_4_8_23p1_.mat'
                # IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_IU_train_1M_8_1_23p1_.mat'
            elif phase == 'val':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_BI_val_2k_4_8_23p1_.mat'
                # IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_IU_val_2k_8_1_23p1_.mat'
            elif phase == 'test':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_BI_test_24k_4_8_23p1_.mat'
                # IU_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_IU_test_24k_8_1_23p1_.mat'
    else:
        raise NameError(f"{channel} is not a valid channel name")

    ### Load channel data
    H_bi = torch.tensor(scio.loadmat(BI_file_path)['H_samples']).to(torch.complex64).to(device)
    # H_iu = torch.tensor(scio.loadmat(IU_file_path)['H_samples']).to(torch.complex64).to(device)

    ### kron the BI and IU parts to get the cascaded channel
    # H_c = batch_khatri_rao(H_bi.permute(0,2,1), H_iu)
    H_c = H_bi
    h_mean = H_c.mean()
    h_std = H_c.std()

    ### Normalize the training data
    if phase == 'train':
        H_c = (H_c - h_mean) / h_std
        # print('Training data')
    h_c = vec(H_c)
    
    ### Signal reflect by the IRS (identity, DFT, or Hadamard)
    # Psi = get_IRS_coef(IRScoef, n_R, n_I, n_T, T).to(device).to(torch.complex64) # IRS coefficient
    x = steering_pilot(n_T_x, n_T_y, device, steering, n_T, T)
    # print(torch.matmul(H_c, x.unsqueeze(1)).shape)
    sgnl = vec(torch.matmul(H_c, x))  # Transmitted signal (vectorized)

    ### Generate noise and received signal
    Ps = (sgnl.abs()**2).mean() # Power of the transmitted signal
    Pn = Ps / SNR_lin # Noise power
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2) # Repeat the noise power for each data sample groups (8)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device) # Generate noise
    y = sgnl + w # Received signal
    return turnReal(h_c), turnReal(y), h_mean, h_std 

def steering_pilot(n_T_x, n_T_y, device, steering, n_T, T):
    if steering.lower() in ['a', 'aligned']:
        x = steering_vector(n_T_x, n_T_y, 0.5, 0.5, torch.tensor(23.1), torch.tensor(0)).to(device).to(torch.complex64) # Steering vector
        x = x.unsqueeze(1) 
        x = x.repeat(1, T)
    elif steering.lower() in ['n', 'null']:
        x = steering_vector(n_T_x, n_T_y, 0.5, 0.5, torch.tensor(-37.4), torch.tensor(0)).to(device).to(torch.complex64) # Steering vector
        x = x.unsqueeze(1) 
        x = x.repeat(1, T)
    elif steering.lower() in ['o', 'omni']:
        if T == 1:
            x = torch.zeros(n_T).to(device).to(torch.complex64) # Steering vector
            x[0] = 1
        elif T == 4:
            x = torch.eye(n_T).to(device).to(torch.complex64)
    else:
        raise NameError(f"{steering} is not a valid steering vector name")



    return x# Return the channel, received signal, mean and std of the channel
