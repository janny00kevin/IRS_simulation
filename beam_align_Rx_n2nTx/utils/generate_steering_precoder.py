import torch
import numpy as np
from utils.steering_vector import steering_vector

def calculate_beam_pattern(steering_vec, n_x, n_y):

    angles_theta = torch.linspace(-90, 90, 180)
    angles_phi = torch.linspace(0, 360, 360)
    angles_phi = torch.linspace(0, 0, 1)
    beam_pattern = torch.zeros((len(angles_theta), len(angles_phi)))
    for i, angle_theta in enumerate(angles_theta):
        for j, angle_phi in enumerate(angles_phi):
            # angle = np.arctan2(np.sin(angle_theta) * np.sin(angle_phi), np.cos(angle_theta))
            array_response = steering_vector(n_x, n_y, 0.5, 0.5, angle_theta, angle_phi)
            beam_pattern[i, j] = 20 * torch.log10(torch.abs(steering_vec.conj() @ array_response.reshape(-1, 1)))

        # array_response = np.exp(1j * np.pi * np.arange(num_antennas) * np.cos(angle))
        # beam_pattern.append(20 * np.log10(np.abs(np.dot(steering_vector, array_response))))
    return beam_pattern

def plot_beam_pattern(beam_pattern):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    beam_pattern = beam_pattern - torch.max(beam_pattern)  # Normalize the beam pattern
    beam_pattern = torch.clamp(beam_pattern, min=-50, max=float('inf')).numpy()

    observation_phi = 0

    angles = np.linspace(-90, 90, beam_pattern.shape[0])
    plt.plot(angles, beam_pattern[:,observation_phi])
    plt.xlabel(r'$\theta$ (degrees)')
    plt.ylabel("Beam Pattern (dB)")
    plt.title("Beam Pattern")
    plt.grid(True)
    plt.ylim(-50, 5)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X = np.linspace(-90, 90, beam_pattern.shape[0])
    # Y = np.linspace(-180, 180, beam_pattern.shape[1])
    # X, Y = np.meshgrid(X, Y)
    # ax.plot_surface(X, Y, beam_pattern.T, cmap='viridis')
    # ax.set_xlabel('Theta')
    # ax.set_ylabel('Phi')
    # ax.set_zlabel('Beam Pattern (dB)')
    # plt.show()

def generate_digital_precoder(num_rf_chains, num_sig_stream, n_x, n_y):
    ### here we assume the number of RF chains is equal to the number of signal streams and the number of antennas ###
    angles_x = torch.tensor(np.linspace(0, 180, n_x, endpoint=False))
    angles_y = torch.tensor(np.linspace(0, 180, n_y, endpoint=False))
    F_bb = torch.zeros((num_rf_chains, num_sig_stream), dtype=complex)
    for x, angle_x in enumerate(angles_x):
        for y, angle_y in enumerate(angles_y):
            i = x * n_y + y
            F_bb[:,i] = steering_vector(n_x, n_y, 0.5, 0.5, angle_x, angle_y)
    return F_bb.to(torch.complex64)

def generate_steering_precoder(n_x, n_y, theta, phi, num_data_stream, device):

    delta_x = 0.5
    delta_y = 0.5
    num_antennas = n_x * n_y

    # generate analog precoder
    f_a = steering_vector(n_x, n_y, delta_x, delta_y, theta, phi)
    F_mat = torch.tile(f_a.reshape(-1, 1), (1, num_data_stream))
    # print(f_a)
    # print(F_mat)

    # generate digital precoder
    # F_bb = generate_digital_precoder(num_rf_chains, num_rf_chains, n_x, n_y)

    # plot_beam_pattern(calculate_beam_pattern((F_rf@F_bb)[:,20], n_x, n_y))

    return F_mat.to(device).to(torch.complex64)
