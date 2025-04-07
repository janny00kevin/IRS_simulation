import torch
import scipy.io as scio
import math
import matplotlib.pyplot as plt
import numpy as np

def steering_vector(n_x, n_y, delta_x, delta_y, theta, phi):
    indices_x = torch.arange(n_x)
    indices_y = torch.arange(n_y)
    phase_x = -2 * math.pi * delta_x * indices_x * torch.sin(torch.deg2rad(theta)) * torch.cos(torch.deg2rad(phi))
    phase_y = -2 * math.pi * delta_y * indices_y * torch.sin(torch.deg2rad(theta)) * torch.sin(torch.deg2rad(phi))
    steering_x = torch.exp(1j * phase_x)
    steering_y = torch.exp(1j * phase_y)
    steering_vec = torch.kron(steering_y, steering_x)
    return steering_vec

# load channel
BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_BI_val_2k_4_8_23p1_.mat'
H_bi = torch.tensor(scio.loadmat(BI_file_path)['H_samples']).to(torch.complex64)
H_bi = H_bi[:100, :, :]
H_bi = torch.squeeze(H_bi[1, :, :])

# define the size of the antenna array
n_T_x = 2
n_T_y = 2
n_R_x = 4
n_R_y = 2
n_T = n_T_x * n_T_y
n_R = n_R_x * n_R_y

delta_x = 0.5
delta_y = 0.5

theta_T_m = torch.tensor(23.1) # matched direction
phi_T = torch.tensor(0)

# analog precoder F_a
steering_vec_T_m = steering_vector(n_T_x, n_T_y, delta_x, delta_y, theta_T_m, phi_T)
F_a_m = torch.diag(steering_vec_T_m)
# signal vec is set to 1
signal_vec = torch.ones(n_T).to(torch.complex64)

# Define angle ranges
theta_range = torch.linspace(0, 90, 181) #仰角範圍
phi = torch.tensor(0) #固定phi為0度

# Calculate beamforming gain
gain = torch.zeros(len(theta_range))

for i, theta in enumerate(theta_range):
    steering_vec = steering_vector(n_T_x, n_T_y, delta_x, delta_y, theta, phi)
    received_signal = torch.matmul(H_bi, torch.matmul(F_a_m, signal_vec))
    power = torch.abs(torch.sum(received_signal))**2
    gain[i] = power

# Visualize the beam pattern
plt.figure()
plt.plot(theta_range.numpy(), gain.numpy())
plt.xlabel('Elevation Angle (theta) [degrees]')
plt.ylabel('Power (abs^2)')
plt.title('Beam Pattern (phi = 0 degrees)')
plt.grid(True)
plt.show()

# Visualize the beam pattern in polar coordinates
theta_rad = torch.deg2rad(theta_range)

plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot(theta_rad.numpy(), gain.numpy())
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.set_xticks(np.pi/180. * np.linspace(0, 90, 5))
ax.set_yticks(np.linspace(gain.min(), gain.max(), 5))
ax.set_title('Beam Pattern (phi = 0 degrees, Polar)')
plt.show()





# phi_sd = 10;
# phi_s = phi_sd*pi/180; 
# lambda = 3e8/28e9;

# d = lambda/2;
# Us = 1/d; 
# U = sin(phi_s)/lambda;
# u = U/Us;

# N=1024;
# w_mf = sqrt(1/n_T_x)*exp(j*2*pi*u*[0:n_T_x-1]');
# W_mf = fftshift(fft(w_mf,N))/sqrt(n_T_x);

# k = -N/2:(N/2-1);
# figure; plot(asind(lambda*k*Us/N), 20*log10(abs(W_mf)));
# title('Beampattern');
# xlabel('angle (deg)'); ylabel('|W_{mf}|^2 (dB)');
# legend('Matched filter');