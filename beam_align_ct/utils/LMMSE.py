import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gen_noisy_sgn(A, SNR, MC):
    # added noise to the matrix A which shape (a, MC) -- MC vectors of a
    # y = A + noise   --> noise corresponding to the SNR in dB
    C_A = cov_mat(A, MC)
    A_pwr = torch.reshape(torch.diag(torch.real(C_A)), (-1, 1))
    noisepwr = A_pwr * 10 ** (-SNR/10)
    C_noise = torch.diag(torch.squeeze(noisepwr))
    C_noise_stdev = torch.linalg.cholesky(C_noise)
    random = torch.complex(torch.randn(A.shape), torch.randn(A.shape)).to(device)
    noise =  (1/2**(1/2)) * C_noise_stdev.to(torch.complex128) @ random.to(torch.complex128)
    # noise = torch.zeros_like(noise).to(torch.complex128)     # no noise SNR -inf
    Y = A + noise
    return Y, C_noise

def cov_mat(A, MC):
    mean_A = torch.reshape(torch.mean(A, dim=1), (-1, 1))
    mean_A_sqr = mean_A @ torch.conj(mean_A.T)
    C_A = 0
    for m in range(MC):
        m1 = torch.reshape(A[:,m], (-1, 1)) 
        m2 = torch.reshape(torch.conj(A[:,m]), (1, -1)) 
        C_A += (m1 @ m2 - mean_A_sqr)
    return C_A / MC

def LMMSE_solver(y, A, x_groundtruth, C_noise, MC):
    # y = A * x  --> estimate for x given A and y
    C_x = cov_mat(x_groundtruth, MC)
    Ax = A @ x_groundtruth
    C_Ax = cov_mat(Ax, MC)
    C_y = C_Ax + C_noise
    C_xy = C_x @ torch.conj(A.T)
    temp = C_xy @ torch.linalg.pinv(C_y)
    mean_x = torch.reshape(torch.mean(x_groundtruth, dim=1), (-1 ,1))
    mean_y = A @ mean_x
    x_lmmse = torch.zeros_like(x_groundtruth)
    for m in range(MC):
        y_ = torch.reshape(y[:,m], (-1,1)).to(torch.complex64)
        x_lmmse[:,m] = torch.squeeze(mean_x + temp @ (y_ - mean_y))
    return x_lmmse

#### USAGE ####
# 1. compute the output of the channel; pilot -> channel -> output
# 2. take the output and add the noise to get the received signal using
#    the function gen_noisy_sgn
# 3. PUt them in the LMMSE_solver