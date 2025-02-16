import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# import LMMSE
import h5py
from utils.batch_kronecker import batch_kronecker
from utils.IRS_rician_channel import importData
from utils.complex_utils import turnReal, turnCplx, vec
from utils.get_IRS_coef import get_IRS_coef
from utils.LMMSE import LMMSE_solver
# from utils.NN_model.MLP_2layer_1024 import MLP
import os

torch.manual_seed(0)

device = torch.device("cuda:1")
script_dir = os.path.dirname(os.path.abspath(__file__))


filename1 = os.path.join(script_dir, 'result', '0.160_SP_ray_MLP_psi_i_lr1e-04_[256, 1024, 258]_ep300.pt')
checkpoint1 = torch.load(filename1, weights_only=False)
filename2 = os.path.join(script_dir, 'result', '0.160_SP_ray_MLP_psi_d_lr1e-04_[256, 1024, 258]_ep150.pt')
checkpoint2 = torch.load(filename2, weights_only=False)
filename3 = os.path.join(script_dir, 'result', '0.160_SP_ray_MLP_psi_h_lr1e-04_[256, 1024, 258]_ep150.pt')
checkpoint3 = torch.load(filename3, weights_only=False)

# h_mean2 = checkpoint2['h_mean'].to(device)
# h_std2 = checkpoint2['h_std'].to(device)
# filename3 = '0.007_SP_elwoRe_UMa_lr1e-04_[288, 3, 256, 1024, 290]_ep14.pt'             ###
# checkpoint3 = torch.load('./simulation/result/model/'+filename3)         #
# filename4 = '0.008_SP_elwRe_UMa_lr1e-04_[288, 3, 256, 1024, 290]_ep17.pt'             ###
# checkpoint4 = torch.load('./simulation/result/model/'+filename4)         #

logits_net1 = checkpoint1['logits_net'].to(device)
logits_net2 = checkpoint2['logits_net'].to(device)
logits_net3 = checkpoint3['logits_net'].to(device) 

# logits_net4 = checkpoint4['logits_net'].to(device)                         #
# n_R, n_T, T = [checkpoint1['n_R'], checkpoint1['n_T'], checkpoint1['T']]
# H_mean, H_sigma, W_mean = [0, 1, 0]

sqrt2 = 2**.5
SNR_dB = torch.arange(-4,10.1,2)
SNR_lin = 10**(SNR_dB/10.0).to(device)
NMSE_1 = torch.zeros_like(SNR_lin) # MLP 1
NMSE_2 = torch.zeros_like(SNR_lin) # MLP 2
NMSE_3 = torch.zeros_like(SNR_lin)
# NMSE_4 = torch.zeros_like(SNR_lin)
# NMSE2 = torch.zeros_like(SNR_lin) # LS
# NMSE4 = torch.zeros_like(SNR_lin) # sampled LMMSE
NMSE_LS_i = torch.zeros_like(SNR_lin) # LS
NMSE_LM_i = torch.zeros_like(SNR_lin) # sampled LMMSE
NMSE_LS_d = torch.zeros_like(SNR_lin) # LS
NMSE_LM_d = torch.zeros_like(SNR_lin) # sampled LMMSE
NMSE_LS_h = torch.zeros_like(SNR_lin) # LS
NMSE_LM_h = torch.zeros_like(SNR_lin) # sampled LMMSE

datasize_per_SNR = int(3e3)
test_data_size = datasize_per_SNR*len(SNR_dB)
# n_R = checkpoint1['n_R']
# n_T = checkpoint1['n_T']
# T = checkpoint1['T']
# n_I = checkpoint1['n_I']
n_R = 4
n_T = 4
T = 32
n_I = 8


# file_path = './simulation/channel_realization/testing_dataset_UMa28.mat'
# with h5py.File(file_path, 'r') as f:
#     h = torch.tensor(f['GroundChan'][:]).T.to(device)
# data_size = h.size(0)
# h = torch.complex(h[:,:144],h[:,144:]).to(torch.complex64)
# h = h
# n_R, n_T, T = 4, 36, 36
# rnd_sample = 1


# X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
# X_tild = torch.complex(torch.eye(n_R*n_T), torch.zeros(n_R*n_T, n_R*n_T)).to(device)
# s = h #@ X_tild 
# print("Ps",Ps.shape)

# H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(data_size, n_R, n_T, 2))/(sqrt2)).to(device)
# Ph = (H.abs()**2).mean()
# print(Ph)

h_test, y_test_i, h_mean, h_std = importData(datasize_per_SNR*len(SNR_lin), n_R, n_I, n_T, T, SNR_lin, device, IRScoef='i', case='test')
_, y_test_d, _, _ = importData(datasize_per_SNR*len(SNR_lin), n_R, n_I, n_T, T, SNR_lin, device, IRScoef='d', case='test')
_, y_test_h, _, _ = importData(datasize_per_SNR*len(SNR_lin), n_R, n_I, n_T, T, SNR_lin, device, IRScoef='h', case='test')
h_test = h_test.reshape(len(SNR_lin), datasize_per_SNR, n_R*n_T*n_I*2)
y_test_i = y_test_i.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
y_test_h = y_test_h.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
y_test_d = y_test_d.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)

pbar = tqdm(total = len(SNR_dB))
# def get_1snr_data(h_test, y_test, datasize_per_SNR, idx):
#     y_1snr = y_test[idx*datasize_per_SNR:(idx+1)*datasize_per_SNR, :]
#     h_1snr = h_test[idx*datasize_per_SNR:(idx+1)*datasize_per_SNR, :]
#     return h_1snr, y_1snr

def LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test, y_test, IRS_coef_type):
    """
    Computes the Least Squares (LS) and Linear Minimum Mean Square Error (LMMSE) norms for given test data.
    Args:
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
        datasize_per_SNR (int): The size of the data per Signal-to-Noise Ratio (SNR).
        n_R (int): Number of receive antennas.
        n_T (int): Number of transmit antennas.
        T (int): Number of time slots.
        n_I (int): Number of IRS elements.
        snr (float): Signal-to-Noise Ratio.
        IRS_coef_type (str): Type of IRS coefficients.
        h_test (torch.Tensor): The test channel matrix.
        y_test (torch.Tensor): The received signal matrix.
    Returns:
        tuple: A tuple containing:
            - norm_LS (torch.Tensor): The norm of the LS estimation error.
            - norm_LM (torch.Tensor): The norm of the LMMSE estimation error.
    """
    Psi = get_IRS_coef(IRS_coef_type,n_R,n_I,n_T,T).to(device)
    tbPsi = batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64)
    ### LS
    # norm_LS = torch.norm((turnCplx(h_test) - turnCplx(y_test)), dim=1)**2
    # if IRS_coef_type ==  'i':
    #     norm_LS = torch.norm((turnCplx(h_test) - turnCplx(y_test)), dim=1)**2
    # elif IRS_coef_type == 'h':
    # Compute the least squares solution
    # pseudo_inv = torch.matmul(torch.linalg.pinv(torch.matmul(tbPsi.T, tbPsi)), tbPsi.T)
    # if snr.item() == 1:
    #     print(torch.linalg.pinv(torch.matmul(tbPsi.T, tbPsi)).shape)
    LS_solution = torch.matmul(tbPsi.pinverse(), turnCplx(y_test).unsqueeze(2)).squeeze(2)
    # LS_solution = tbPsi.pinverse() @ turnCplx(y_test).unsqueeze(2)
    # Compute the norm of the difference
    norm_LS = torch.norm((turnCplx(h_test) - LS_solution), dim=1)**2
    # else:
    #     raise ValueError("IRS coefficient matrix should be 'identity'('i') or 'hadamard'('h')")
    # norm2 = torch.norm((turnCplx(h_test) - torch.matmul(tbPsi.T, turnCplx(y_test).unsqueeze(2)).squeeze(2)/tbPsi.shape[0]), dim=1)**2

    ### LMMSE
    tbPsi = batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64)
    Sgnl = torch.matmul(tbPsi, turnCplx(h_test).unsqueeze(2)).squeeze(2)
    Ps = (Sgnl.abs()**2).mean(dim=0)
    Pn = Ps / snr
    cov_n = torch.diag(Pn)
    lmmse = LMMSE_solver(turnCplx(y_test).T, tbPsi, turnCplx(h_test).T, cov_n, datasize_per_SNR).T
    norm_LM = torch.norm((turnCplx(h_test) - lmmse), dim=1)**2
    return norm_LS,norm_LM

for idx, snr in enumerate(SNR_lin):
    snr = snr.unsqueeze(0).to(device)

    D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

    # IRS_coef_type = checkpoint1['IRS_coe_type']

    # # Y_nmlz = (y.reshape(data_size,n_R,T)-h_mean2)/h_std2
    logits1 = logits_net1(turnReal(turnCplx(y_test_i[idx])-h_mean)/h_std)
    test_tbh_cplx = turnCplx(logits1)*h_std + h_mean
    norm_1 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

    logits2 = logits_net2(turnReal(turnCplx(y_test_d[idx])-h_mean)/h_std)
    test_tbh_cplx = turnCplx(logits2)*h_std + h_mean
    norm_2 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

    logits3 = logits_net3(turnReal(turnCplx(y_test_h[idx])-h_mean)/h_std)
    test_tbh_cplx = turnCplx(logits3)*h_std + h_mean
    norm_3 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2


    '''
    LS and LMMSE numerical solution for different IRS coefficient matrix
    '''

    norm_LS_i, norm_LM_i = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_i[idx], IRS_coef_type='i')
    norm_LS_d, norm_LM_d = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_d[idx], IRS_coef_type='d')
    norm_LS_h, norm_LM_h = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_h[idx], IRS_coef_type='h')

    # logits3 = logits_net3(torch.stack([Y_nmlz.real,Y_nmlz.imag,Y_nmlz.abs()],dim=1))
    # test_tbh_cplx = turnCplx(logits3)
    # norm_3 = torch.norm(h - D.matmul(test_tbh_cplx.T).T*h_std2-h_mean2, dim=1)**2
    # logits4 = logits_net4(torch.stack([Y_nmlz.real,Y_nmlz.imag,Y_nmlz.abs()],dim=1))
    # test_tbh_cplx = turnCplx(logits4)
    # norm_4 = torch.norm(h - D.matmul(test_tbh_cplx.T).T*h_std2-h_mean2, dim=1)**2


    with torch.no_grad():
        NMSE_1[idx] = 10*torch.log10((norm_1 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_2[idx] = 10*torch.log10((norm_2 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_3[idx] = 10*torch.log10((norm_3 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_4[idx] = 10*torch.log10((norm_4 / torch.norm((h), dim=1)**2).mean()) #

    NMSE_LS_i[idx] = 10*torch.log10((norm_LS_i / torch.norm(h_test[idx], dim=1)**2).mean())
    NMSE_LM_i[idx] = 10*torch.log10((norm_LM_i / torch.norm(h_test[idx], dim=1)**2).mean())
    NMSE_LS_d[idx] = 10*torch.log10((norm_LS_d / torch.norm(h_test[idx], dim=1)**2).mean())
    NMSE_LM_d[idx] = 10*torch.log10((norm_LM_d / torch.norm(h_test[idx], dim=1)**2).mean())
    NMSE_LS_h[idx] = 10*torch.log10((norm_LS_h / torch.norm(h_test[idx], dim=1)**2).mean())
    NMSE_LM_h[idx] = 10*torch.log10((norm_LM_h / torch.norm(h_test[idx], dim=1)**2).mean())
    
    # plt.text(10*torch.log10(snr)-1,NMSE_LM_i[idx]-1, f'({NMSE_LM_i[idx].item():.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr)-1,NMSE_2[idx]-1, f'({10**(NMSE_2[idx].item()/10):.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr),NMSE_3[idx], f'({10**(NMSE_3[idx].item()/10):.2f})')  ## plot linear NMSE value 
    pbar.update(1)

    

plt.plot(SNR_dB, NMSE_LS_i.to('cpu'), label='LS with identity', linewidth=1, linestyle='--', marker='o', color="tab:blue")
plt.plot(SNR_dB, NMSE_LS_d.to('cpu'), label='LS with DFT', linewidth=1, linestyle=':', marker='o', color="tab:blue")
plt.plot(SNR_dB, NMSE_LS_h.to('cpu'), label='LS with Hadamard', linewidth=1, linestyle='-', marker='o', color="tab:blue")
plt.plot(SNR_dB, NMSE_LM_i.to('cpu'), label='LMMSE with identity', linewidth=1, linestyle='--', marker='o', color="tab:red")
plt.plot(SNR_dB, NMSE_LM_d.to('cpu'), label='LMMSE with DFT', linewidth=1, linestyle=':', marker='o', color="tab:red")
plt.plot(SNR_dB, NMSE_LM_h.to('cpu'), label='LMMSE with Hadamard', linewidth=1, linestyle='-', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE5,'-x', label='model-based PD LS', linewidth=1 ,color="tab:red")
# plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')

plt.plot(SNR_dB, NMSE_1.to('cpu'), label='NP PD with identity', linewidth=1, linestyle='--', marker='x', color="tab:orange")   ###
plt.plot(SNR_dB, NMSE_2.to('cpu'), label='NP PD with DFT', linewidth=1, linestyle=':', marker='x', color="tab:orange")  ###
plt.plot(SNR_dB, NMSE_3.to('cpu'), label='NP PD with Hadamard', linewidth=1, linestyle='-', marker='x', color="tab:orange")  ###

# plt.plot(SNR_dB, NMSE_3,'-x', label='channelNet w/o act. func.')  ###
# plt.plot(SNR_dB, NMSE_4,'-x', label='channelNet w/ act. func.')  ###

plt.suptitle("NP PD trained MMSE MIMO nmlz ChEst with IRS vs SNR in Rician")
# plt.title(' $[n_R,n_T]$:[4,36], MLP size:[288, 2048, 2048, 576] ')

# plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title('size:%s' %([2*n_R*T]+checkpoint1['hidden_sizes']+[2*n_R*n_T]))
plt.title('$[n_R,n_I,n_T,T]:[%s,%s,%s,%s], datasize: %s$' %(n_R,n_I,n_T,T,datasize_per_SNR))
plt.xlabel('SNR(dB)')
plt.ylabel('NMSE(dB)')
plt.legend()
plt.grid(True)
# plt.savefig('./simulation/result/snr/LS_mlp_Chest_-4_vs_i2.png')   ###
save_path = os.path.join(script_dir, 'snr', 'SP_ric_idh_MLP_10M_nmlzdata.pdf')#  %(IRS_coef_type)
plt.savefig(save_path)   ###

