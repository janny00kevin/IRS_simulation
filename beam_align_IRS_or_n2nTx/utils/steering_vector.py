import torch
import math

def steering_vector(n_x, n_y, delta_x, delta_y, theta, phi):
    """
    Generates a steering vector for a 2D array.

    Args:
        n_x: Number of elements in the x-axis.
        n_y: Number of elements in the y-axis.
        delta_x: Distance between elements in the x-axis.
        delta_y: Distance between elements in the y-axis.
        theta: Elevation angle.
        phi: Azimuth angle.
    """
    indices_x = torch.arange(n_x)
    indices_y = torch.arange(n_y)
    phase_x = -2 * math.pi * delta_x * indices_x * torch.sin(torch.deg2rad(theta)) * torch.cos(torch.deg2rad(phi))
    phase_y = -2 * math.pi * delta_y * indices_y * torch.sin(torch.deg2rad(theta)) * torch.sin(torch.deg2rad(phi))
    steering_x = torch.exp(1j * phase_x)
    steering_y = torch.exp(1j * phase_y)
    steering_vec = torch.kron(steering_y, steering_x)
    return steering_vec