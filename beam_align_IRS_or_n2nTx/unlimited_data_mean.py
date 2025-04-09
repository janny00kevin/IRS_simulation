import torch
from utils.IRS_ct_channels_align import importData
train_size = 1e6
n_T, n_I, n_R, T = 4, 8, 1, 32
device = 'cuda:1'
snr = [-4, 10]
SNR_dB = torch.tensor(list(range(min(snr),max(snr)+1,2))).to(device)   ###
SNR_lin = 10**(SNR_dB/10.0)
IRS_coe_type = 'h'
channel = 'UMa'
angle_config = 'o'

h, y, h_mean, h_std = importData(train_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type, phase='train', channel=channel, config=angle_config)

print(h[10000,1,1])


'''
the mean of the cascaded channel H_c is about 10^-9, b/c it's  kron product of two matrices, BI (10^-4) and IU (10^-3).

'''