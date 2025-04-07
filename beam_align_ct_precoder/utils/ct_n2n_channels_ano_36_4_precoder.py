import torch
# import scipy.io as scio
import h5py
from utils.complex_utils import turnReal, vec
from utils.batch_khatri_rao import batch_khatri_rao
# from utils.get_IRS_coef import get_IRS_coef
from utils.steering_vector import steering_vector
from utils.generate_steering_precoder import generate_steering_precoder

def importData(data_size, n_R_x, n_R_y, n_T_x, n_T_y, T, SNR_lin, device, phase = 'train', channel='x', steering='x'):
    """
    Imports and processes channel data for N2N simulation.

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
                BI_file_path = './IRS_simulation/beam_align_ct_precoder/channels_dataset/uma_BI_train_1M_36_4_23p1_.mat'
            elif phase == 'val':
                BI_file_path = './IRS_simulation/beam_align_ct_precoder/channels_dataset/uma_BI_val_2k_36_4_23p1_.mat'
            elif phase == 'test':
                BI_file_path = './IRS_simulation/beam_align_ct_precoder/channels_dataset/uma_BI_test_24k_36_4_23p1_.mat'
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
    # H_bi = torch.tensor(scio.loadmat(BI_file_path)['H_samples']).to(torch.complex64).to(device)
    with h5py.File(BI_file_path, 'r') as f:
        H_samples_real = f['H_samples']['real'][()]
        H_samples_imag = f['H_samples']['imag'][()]
        # H_samples_complex =  
        H_bi = torch.tensor(H_samples_real + 1j * H_samples_imag, dtype=torch.complex64).permute(2,1,0).to(device)
    # print("H_bi", H_bi.shape)

    ### kron the BI and IU parts to get the cascaded channel
    h_mean = H_bi.mean()
    h_std = H_bi.std()

    ### Normalize the training data
    if phase == 'train':
        H_bi = (H_bi - h_mean) / h_std
    h = vec(H_bi)
    
    ### Signal reflect by the IRS (identity, DFT, or Hadamard)
    # x = steering_pilot(n_T_x, n_T_y, device, steering, n_T, T)
    # sgnl = vec(torch.matmul(H_bi, x))  # Transmitted signal (vectorized)
    # s = torch.eye(n_T).to(device).to(torch.complex64) # Transmitted signal (identity matrix)

    ### steering precoder
    # F_rf, F_bb = generate_steering_precoder(n_T_x, n_T_y, torch.tensor(23.1), torch.tensor(0), n_T_x*n_T_y, device)
    X_mat = steering_pilot(n_T_x, n_T_y, device, steering, n_T, T)
    sgnl_channel = vec(H_bi @ X_mat) # Transmitted signal (vectorized)
    

    ### Generate noise and received signal
    Ps = (sgnl_channel.abs()**2).mean() # Power of the transmitted signal
    Pn = Ps / SNR_lin # Noise power
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2) # Repeat the noise power for each data sample groups (8)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device) # Generate noise
    y_ant = sgnl_channel + w # Received signal
    # print(y.shape)

    # ### steering the Rx beam
    # f_a = torch.flip(steering_vector(n_R_x, n_R_y, 0.5, 0.5, torch.tensor(23.1), torch.tensor(0)).to(device).to(torch.complex64), dims=[0])
    # # f_a = torch.ones(n_R).to(device).to(torch.complex64)
    # F_a = torch.diag(f_a.conj())
    # tF_a = torch.kron(torch.eye(T).to(device), F_a)

    ### steering Rx combiner
    if steering == 'a' or steering == 'aligned':
        W_rf, W_bb = generate_steering_precoder(n_R_x, n_R_y, torch.tensor(23.1), torch.tensor(0), n_R_x*n_R_y, device)
    elif steering == 'n' or steering == 'null':
        W_rf, W_bb = generate_steering_precoder(n_R_x, n_R_y, torch.tensor(3.2), torch.tensor(0), n_R_x*n_R_y, device)
    elif steering == 'o' or steering == 'omni':
        W_rf = torch.eye(n_R).to(device).to(torch.complex64)
        W_bb = torch.eye(n_R).to(device).to(torch.complex64)
    kron_combiner = torch.kron(torch.eye(T).to(device), W_bb.conj().T @ W_rf.conj().T)
    y = (kron_combiner.unsqueeze(0) @ y_ant.unsqueeze(-1)).squeeze(-1) # Received signal (vectorized)
    # print(torch.isnan(y))
    # print((y.abs()**2).mean())
    # y = (tF_a.unsqueeze(0) @ y.unsqueeze(-1)).squeeze(-1)
    # print(y.shape)

    return turnReal(h), turnReal(y), h_mean, h_std 

def steering_pilot(n_T_x, n_T_y, device, steering, n_T, T):
    Sgnl = torch.eye(n_T).to(device).to(torch.complex64)
    match steering.lower():
        case 'a' | 'aligned':
            F_rf, F_bb = generate_steering_precoder(n_T_x, n_T_y, torch.tensor(23.1), torch.tensor(0), n_T_x*n_T_y, device)
            X_mat = F_rf @ F_bb @ Sgnl
        case 'n' | 'null':
            F_rf, F_bb = generate_steering_precoder(n_T_x, n_T_y, torch.tensor(3.2), torch.tensor(0), n_T_x*n_T_y, device)
            X_mat = F_rf @ F_bb @ Sgnl
        case 'o' | 'omni':
            X_mat = Sgnl
        case _:
            raise NameError(f"{steering} is not a valid steering vector name")
    
    return X_mat
