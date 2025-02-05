from utils.batch_khatri_rao import batch_khatri_rao  # Import the function
import torch
 
'''
Khatri-Rao product
'''
# # Example usage
# data_size = 10
# n_ax, n_ay, n_bx, n_by = 3, 4, 5, 4  # Ensure n_ay == n_by
# H_a = torch.randn(data_size, n_ax, n_ay, dtype=torch.complex64)
# H_b = torch.randn(data_size, n_bx, n_by, dtype=torch.complex64)
# # Define A and B
# A = torch.tensor([[[1, 2], [3, 4]],[[1, 3], [2, 4]]])  # Shape: (2, 2, 2)
# B = torch.tensor([[[5, 6], [7, 8], [9, 10]],[[2, 7], [3, 2], [4, 1]]])  # Shape: (2, 3, 2)

# # Compute Khatri-Rao product
# result = batch_khatri_rao(A, B)
# # print(result)

'''
matrix vec
'''
# d,m,n = 3,4,2
# a = torch.arange(d*m*n)
# a = a.reshape(d,m,n)
# print(a)
# print(a.permute(0, 2, 1).reshape(d,m*n,1))


'''
Pn repeat_interleave
'''
# Pn = torch.tensor([1.0,0.5])
# data_size, n_R, T = 2, 3, 4
# Pn = Pn.repeat_interleave(2*n_R*T*2//len(Pn)).reshape(data_size, n_R*T, 2)
# print(Pn)


'''
test IRS_rayleigh_channel.py
'''
# from utils.IRS_rayleigh_channel import importData
# SNR_lin = torch.tensor([0]).to('cuda')
# h, y, h_mean, h_std = importData(data_size=2, n_R=2, n_I=4, n_T=2, T=8, SNR_lin=SNR_lin, device='cuda', W_Mean=0, IRScoef='r')

'''
test LS pesudo inverse
'''
from utils.get_IRS_coef import get_IRS_coef
psi = get_IRS_coef('identity', 2, 4, 2, 8)
print(psi.shape)
print(torch.matmul(psi.T, psi))