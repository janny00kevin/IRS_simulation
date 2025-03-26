import torch
import scipy.io as scio
import math
import matplotlib.pyplot as plt

def steering_vector(n_x, n_y, delta_x, delta_y, theta, phi):
    indices_x = torch.arange(n_x)
    indices_y = torch.arange(n_y)
    phase_x = -2 * math.pi * delta_x * indices_x * torch.sin(torch.deg2rad(theta)) * torch.cos(torch.deg2rad(phi))
    phase_y = -2 * math.pi * delta_y * indices_y * torch.sin(torch.deg2rad(theta)) * torch.sin(torch.deg2rad(phi))
    steering_x = torch.exp(1j * phase_x)
    steering_y = torch.exp(1j * phase_y)
    steering_vec = torch.kron(steering_y, steering_x)
    return steering_vec

# device = torch.device('cuda:0')

# load channel
BI_file_path = './IRS_simulation/beam_align_ct/channels/angle_23p1/UMa_BI_val_2k_4_8_23p1_.mat'
H_bi = torch.tensor(scio.loadmat(BI_file_path)['H_samples']).to(torch.complex64)
H_bi = H_bi[:100, :, :]
# H_bi = torch.squeeze(H_bi[1, :, :])

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
theta_T_o = torch.tensor(-37.4) # theta_T_m - 180 # orthogonal direction
phi_T = torch.tensor(0)

# analog precoder F_a
steering_vec_T_m = steering_vector(n_T_x, n_T_y, delta_x, delta_y, theta_T_m, phi_T)
steering_vec_T_o = steering_vector(n_T_x, n_T_y, delta_x, delta_y, theta_T_o, phi_T)
f_a_s = torch.zeros(n_T).to(torch.complex64)
f_a_s[2] = 1  * (n_T)**0.5
print(steering_vec_T_m.norm(), steering_vec_T_o.norm(), f_a_s.norm())

F_a_m = torch.diag(steering_vec_T_m)
F_a_o = torch.diag(steering_vec_T_o)
F_a_s = torch.diag(f_a_s)

# signal vec is set to 1
signal_vec = torch.ones(n_T).to(torch.complex64)

theta_R_values = torch.arange(-90, 91, 1)
phi_R_values = torch.tensor([0])

# print(theta_R_values[0], theta_R_values[-1], theta_R_values.shape)

all_signal_powers = torch.zeros((H_bi.shape[0], 3, len(theta_R_values), len(phi_R_values)))  # Store signal powers for all samples
# all_theta_R_lists = []  # Store receive angles for all samples
# print(torch.zeros((len(theta_R_values), len(phi_R_values))).shape)

# indices_x_R = torch.arange(n_R_x)
# indices_y_R = torch.arange(n_R_y)
for itr, H_bi_sample in enumerate(H_bi):
    signal_powers = torch.zeros((len(theta_R_values), len(phi_R_values)))  # Store signal powers for each sample
    # theta_R_list = []  # Store receive angles for each sample
    # received signal
    for num_Fa_type, F_a in enumerate([F_a_m, F_a_o, F_a_s]):
        y = H_bi_sample @ F_a @ signal_vec
        for i, theta_R in enumerate(theta_R_values):
            for j, phi_R in enumerate(phi_R_values):
                # Calculate receive steering vector
                # print(steering_vector(n_R_x, n_R_y, delta_x, delta_y, theta_R, phi_R).shape)
                steering_vec_R = steering_vector(n_R_x, n_R_y, delta_x, delta_y, theta_R, phi_R)

                # Receive beamforming from diff angles
                received_signal = steering_vec_R.conj().reshape(1, -1) @ y

                # Calculate signal power
                signal_power = torch.abs(received_signal) ** 2

                signal_powers[i, j] = signal_power.item()
                # theta_R_list.append(theta_R.item())
        all_signal_powers[itr, num_Fa_type, :, :] = signal_powers
    # all_theta_R_lists.append(theta_R_list)
# print("all_signal_powers.shape",len(all_signal_powers))
# print(all_signal_powers.shape)
all_signal_powers = torch.mean(all_signal_powers, dim=0)
all_signal_powers_dB = 10 * torch.log10(all_signal_powers.reshape(3, len(theta_R_values)))
max_value = torch.max(all_signal_powers_dB).item()
# print(max_value)
# print(all_signal_powers.shape)

# Plot the graph
plt.plot(theta_R_values.numpy(), all_signal_powers_dB[0,:].numpy() - max_value, label='Tx steering to Matched Direction')
plt.plot(theta_R_values.numpy(), all_signal_powers_dB[1,:].numpy() - max_value, label=f'Tx steering to {theta_T_o:.1f}$^\\circ$')
plt.plot(theta_R_values.numpy(), all_signal_powers_dB[2,:].numpy() - max_value, label='Omni-directional')
plt.xlabel(r"Receive Angle ($\theta_R$)")
plt.ylabel("Received signal Power $|\\bf{f}_{a,r}^H \\bf{y}|^2$  (dB)")
plt.title("Receive Beam Response through UMa 28 GHz Channel")
plt.legend()
plt.grid(True)
plt.show()