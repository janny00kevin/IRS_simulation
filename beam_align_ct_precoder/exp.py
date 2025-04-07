import torch
from utils.generate_steering_precoder import generate_steering_precoder, calculate_beam_pattern, plot_beam_pattern
from utils.ct_n2n_channels_ano_36_4_precoder import importData, steering_pilot


# generate_steering_precoder(6, 6, torch.tensor(23.1), torch.tensor(0), 36)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# SNR_dB = torch.tensor(list(range(-4, 10+1, 2))).to(device)   ###
# SNR_dB = torch.tensor([-100]).to(device)
# SNR_lin = 10**(SNR_dB/10.0)
# h, y, h_mean, h_std = importData(int(2e3), 2, 2, 6, 6, 36, SNR_lin, device, 'val', 'UMa', 'o')

X_mat = steering_pilot(6,6,device,'a',36,36)
plot_beam_pattern(calculate_beam_pattern(X_mat[:,16], 6, 6))
