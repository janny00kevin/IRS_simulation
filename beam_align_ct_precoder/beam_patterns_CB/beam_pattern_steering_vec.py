import numpy as np
import matplotlib.pyplot as plt

def generate_steering_vector(num_antennas, angle):
    """
    Generates a steering vector.

    Parameters:
        num_antennas (int): Number of antennas.
        angle (float): Incident angle in degrees.

    Returns:
        numpy.ndarray: Steering vector.
    """
    n = np.arange(num_antennas)
    steering_vector = np.exp(-1j  * np.pi  * n * np.cos(angle * np.pi / 180))
    return steering_vector

def generate_digital_precoder(num_rf_chains, num_sig_stream):
    angles = np.linspace(0, 180, num_sig_stream, endpoint=False)
    F_bb = np.zeros((num_rf_chains, num_sig_stream), dtype=complex)
    for i, angle in enumerate(angles):
        F_bb[:,i] = generate_steering_vector(num_rf_chains, angle)
    return F_bb

def calculate_beam_pattern(steering_vector, angles, num_antennas):
    """
    Calculates the beam pattern.

    Parameters:
        steering_vector (numpy.ndarray): Steering vector.
        angles (numpy.ndarray): Angles at which to calculate the beam pattern.
        num_antennas (int) : number of antennas
        wavenumber (float) : wavenumber

    Returns:
        numpy.ndarray: Beam pattern.
    """

    beam_pattern = []
    for angle in angles:
        array_response = np.exp(1j * np.pi * np.arange(num_antennas) * np.cos(angle))
        beam_pattern.append(20 * np.log10(np.abs(np.dot(steering_vector, array_response))))
    return np.array(beam_pattern)

# Parameter settings
num_antennas = 6  # Number of antennas
steer_angle = 23.1  # Incident angle cosine value
# wavenumber = 0.5  # Wavenumber
angles = np.linspace(-np.pi/2, np.pi/2, 360)  # Angles to calculate the beam pattern
num_rf_chains = num_antennas # Number of RF chains

# Generate steering vectors
f_rf_1 = generate_steering_vector(num_antennas, steer_angle)
# print(steering_vector.shape)
# steering_vector2 = generate_steering_vector(num_antennas, 25 + 11.25, wavenumber)
# shift = np.exp(1j * 0.3 * np.pi)
# steering_vector = steering_vector1 #+ shift * steering_vector2
# steering_vector = steering_vector1

#
F_rf = np.tile(f_rf_1.reshape(-1, 1), num_rf_chains)

s = np.zeros(num_rf_chains, dtype=complex)
s[1] = 1
# print(s)

# angles = np.linspace(0, 180, num_rf_chains, endpoint=False)
# print(angles)

# Generate digital precoder
F_bb = generate_digital_precoder(num_rf_chains, num_rf_chains)

s_precoded = F_rf @ F_bb @ s


# Calculate beam pattern
beam_pattern = calculate_beam_pattern(s_precoded, angles, num_antennas)

# Plot beam pattern
plt.plot(np.rad2deg(angles), (beam_pattern - np.max(beam_pattern)))
plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized Magnitude (dB)")
plt.title("Beam Pattern")
plt.grid(True)
plt.ylim(-50, 5)
plt.show()