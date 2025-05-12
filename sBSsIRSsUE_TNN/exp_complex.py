import torch

# # 創建一個複數張量，這裡使用 torch.complex64 類型
# a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
# b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)

# # 讓複數張量的實部是 `a`，虛部是 `b`
# z = torch.complex(a, b)  # 複數張量

# # 開啟微分
# z.requires_grad_()

# # 定義一個簡單的複數函數
# y = torch.abs(z)**2  # 這是 |z|^2，對應於 z*z^*，即 (實部^2 + 虛部^2)

# # 計算 y 對 z 的梯度
# y.sum().backward()  # 計算總和的反向傳播

# # 打印梯度，這將顯示 z 的實部和虛部對應的梯度
# print("z:",z.grad)

###
# a = torch.arange(6).reshape(2,3)
# b = torch.arange(30).reshape(10,3)

# print("a: ",a)
# print("b: ",b[0,:])
# print(torch.matmul(a.unsqueeze(0),b.unsqueeze(-1)).squeeze()[0,:])


######
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# import LMMSE
import h5py
from utils.batch_kronecker import batch_kronecker
from utils.IRS_ct_channels import importData
from utils.complex_utils import turnReal, turnCplx, vec
from utils.get_IRS_coef import get_IRS_coef
from utils.LMMSE import LMMSE_solver
# from utils.NN_model.MLP_2layer_1024 import MLP
from utils.NN_model.ISTANet import ISTANet
import os

# torch.manual_seed(0)

# device = torch.device("cuda:1")
# device1 = torch.device("cuda:2")
# script_dir = os.path.dirname(os.path.abspath(__file__))
# n_T = 4
# n_I = 8
# n_R = 4
# T = 32

# # MLPs
# filename1 = os.path.join(script_dir, 'trained_model', '0.790_SP_uma_MLP_TNN_lr1e-03_[256, 1024, 258]_ep92.pt')
# checkpoint1 = torch.load(filename1, weights_only=False)

# Psi = checkpoint1['Psi'].to(device)
# print(Psi)
# print(Psi.conj().T.matmul(Psi).abs())
# print(Psi.shape)

# from utils.get_IRS_coef import get_IRS_coef
# Psi = get_IRS_coef('d',4,8,4,32).to(device)
# print(Psi)
# print(Psi.conj().T.matmul(Psi).abs())
# print(Psi.shape)


# import torch

# # 創建一個 8x8 的複數矩陣
# Psi = torch.randn(8, 8, dtype=torch.cfloat)

# # 計算 Psi^H * Psi
# Psi_H_Psi = Psi.conj().T.matmul(Psi)

# # 顯示結果
# print(Psi_H_Psi)


# #####
# from utils.IRS_ct_channels_TNN import importData, add_noise
# train_size = int(1e6) # 1e6
# device = torch.device("cuda:1")
# channel = 'uma'


# ## load training and testing data
# h, h_mean, h_std = importData(train_size, device, phase = 'train', channel=channel)
# h_test, _, _ = importData(int(2.4e4), device, phase = 'test', channel=channel)





#######
# import torch
# import time

# Pn = torch.ones(1,int(1e9))

# while True:
#     w = (torch.normal(0, Pn)/2).to('cuda:1')
#     time.sleep(3000)


####
import torch

a = torch.Pn = torch.tensor([3.8879e-06, 2.4531e-06, 1.5478e-06, 9.7660e-07, 6.1619e-07, 3.8879e-07, 2.4531e-07, 1.5478e-07])
milliwatt_power = a * 1000 * 10**(-2)
dBm = 10 * torch.log10(milliwatt_power)
print(dBm)