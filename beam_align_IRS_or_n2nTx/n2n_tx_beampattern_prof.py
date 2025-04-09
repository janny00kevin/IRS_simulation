import numpy as np
import matplotlib.pyplot as plt

theta_sd = -37.4
theta_s = theta_sd * np.pi / 180
lambda_val = 3e8 / 28e9
## assume phi_s = 0, then cos(phi_s) = 1

d = lambda_val / 2
Us = 1 / d
U = np.sin(theta_s) / lambda_val
u = U / Us

n_T_x = 2
N = 1024
w_mf = np.sqrt(1 / n_T_x) * np.exp(1j * 2 * np.pi * u * np.arange(n_T_x))
W_mf = np.fft.fftshift(np.fft.fft(w_mf, N)) / np.sqrt(n_T_x)

k = np.arange(-N / 2, N / 2)
angle = np.arcsin(lambda_val * k * Us / N) * 180 / np.pi  # Convert radians to degrees

W_mf_dB = 20 * np.log10(np.abs(W_mf))


plt.figure()
plt.plot(angle, W_mf_dB)

min_y = np.min(W_mf_dB)
min_x_index = np.argmin(W_mf_dB)
min_x = angle[min_x_index]
max_y = np.max(W_mf_dB)
max_x_index = np.argmax(W_mf_dB)
max_x = angle[max_x_index]

plt.scatter(min_x, min_y, color='red', marker='o')  
plt.text(min_x, min_y, f'Min: ({min_x:.1f}$^\\circ$, {min_y:.2f})', verticalalignment='bottom', horizontalalignment='right') 
plt.scatter(max_x, max_y, color='green', marker='o')
plt.text(max_x, max_y, f'Max: ({max_x:.1f}$^\\circ$, {max_y:.2f})', verticalalignment='bottom', horizontalalignment='left')

plt.title(f'Beam pattern of Tx pointing to $\\phi = {theta_sd:.1f}^\\circ$ (antenna x-num: {n_T_x}, $f_c$ = 28 GHz)')
plt.xlabel('angle ($\\theta_T$)')
plt.ylabel('$|\\bf{f}_a s|^2$ (dB)')
# plt.legend(['Matched filter'])
plt.show()