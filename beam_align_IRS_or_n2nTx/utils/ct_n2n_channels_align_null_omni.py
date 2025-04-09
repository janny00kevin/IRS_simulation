import torch
import scipy.io as scio
from utils.complex_utils import turnReal, vec
from utils.batch_khatri_rao import batch_khatri_rao
# from utils.get_IRS_coef import get_IRS_coef
from utils.steering_vector import steering_vector

def importData(data_size, n_R_x, n_R_y, n_T_x, n_T_y, T, SNR_lin, device, phase = 'train', channel='x', steering='x'):
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
    n_R = n_R_x * n_R_y

    ## generate communication data to train the parameterized policy
    if channel.lower() == 'uma':
            if phase == 'train':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_BI_train_1M_4_8_23p1_.mat'
            elif phase == 'val':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_BI_val_2k_4_8_23p1_.mat'
            elif phase == 'test':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_BI_test_24k_4_8_23p1_.mat'
    elif channel.lower() == 'inf':
            if phase == 'train':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_BI_train_1M_4_8_23p1_.mat'
            elif phase == 'val':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_BI_val_2k_4_8_23p1_.mat'
            elif phase == 'test':
                BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/InF_BI_test_24k_4_8_23p1_.mat'
    else:
        raise NameError(f"{channel} is not a valid channel name")

    ### Load channel data
    H_bi = torch.tensor(scio.loadmat(BI_file_path)['H_samples']).to(torch.complex64).to(device)

    ### kron the BI and IU parts to get the cascaded channel
    h_mean = H_bi.mean()
    h_std = H_bi.std()

    ### Normalize the training data
    if phase == 'train':
        H_bi = (H_bi - h_mean) / h_std
    h = vec(H_bi)
    
    ### Signal reflect by the IRS (identity, DFT, or Hadamard)
    x = steering_pilot(n_T_x, n_T_y, device, steering, n_T, T)
    sgnl = vec(torch.matmul(H_bi, x))  # Transmitted signal (vectorized)

    ### Generate noise and received signal
    Ps = (sgnl.abs()**2).mean() # Power of the transmitted signal
    Pn = Ps / SNR_lin # Noise power
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2) # Repeat the noise power for each data sample groups (8)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device) # Generate noise
    y = sgnl + w # Received signal
    # print(y.shape)

    # ### steering the Rx beam
    f_a = torch.flip(steering_vector(n_R_x, n_R_y, 0.5, 0.5, torch.tensor(23.1), torch.tensor(0)).to(device).to(torch.complex64), dims=[0])
    # f_a = torch.ones(n_R).to(device).to(torch.complex64)
    F_a = torch.diag(f_a.conj())
    tF_a = torch.kron(torch.eye(T).to(device), F_a)
    y = (tF_a.unsqueeze(0) @ y.unsqueeze(-1)).squeeze(-1)
    # print(y.shape)

    return turnReal(h), turnReal(y), h_mean, h_std 

def steering_pilot(n_T_x, n_T_y, device, steering, n_T, T):
    match steering.lower():
        case 'a' | 'aligned':
            x = steering_vector(n_T_x, n_T_y, 0.5, 0.5, torch.tensor(23.1), torch.tensor(0)).to(device).to(torch.complex64) # Steering vector
            X_mat = x.unsqueeze(1) .repeat(1, T)
        case 'n' | 'null':
            x = steering_vector(n_T_x, n_T_y, 0.5, 0.5, torch.tensor(-37.4), torch.tensor(0)).to(device).to(torch.complex64) # Steering vector
            X_mat = x.unsqueeze(1).repeat(1, T)
        case 'o' | 'omni':
            if T == 1:
                x = torch.zeros(n_T).to(device).to(torch.complex64) # Steering vector
                x[0] = 1
            elif T == 4:
                X_mat = torch.eye(n_T).to(device).to(torch.complex64)
                # x = torch.zeros(n_T).to(device).to(torch.complex64) # Steering vector
                # x[0] = 1
                # X_mat = x.unsqueeze(1) .repeat(1, T)
        case _:
            raise NameError(f"{steering} is not a valid steering vector name")
    
    return X_mat
