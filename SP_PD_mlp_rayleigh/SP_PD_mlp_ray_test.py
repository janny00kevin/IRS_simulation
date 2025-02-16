import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# import LMMSE
import h5py
from utils.batch_kronecker import batch_kronecker
from utils.IRS_rayleigh_channel import importData
from utils.complex_utils import turnReal, turnCplx, vec
from utils.get_IRS_coef import get_IRS_coef
from utils.LMMSE import LMMSE_solver
import os

# torch.manual_seed(0)

device = torch.device("cuda:1")
script_dir = os.path.dirname(os.path.abspath(__file__))

# filename1 = '0.007_SPc_2ch_UMa_lr1e-04_[288, 3, 8, 1024, 290]_ep533.pt'   ###
filename1 = os.path.join(script_dir, 'result', '0.443_SP_ray_psi_i_lr1e-05_[256, 1024, 1024, 258]_ep300.pt')
checkpoint1 = torch.load(filename1, weights_only=False)
# filename2 = '0.005_SP_UMa_nmlz_lr1e-04_[288, 1024, 1024, 290]_ep350.pt'    ###
# checkpoint2 = torch.load('./simulation/result/model/'+filename2)
# h_mean2 = checkpoint2['h_mean'].to(device)
# h_std2 = checkpoint2['h_std'].to(device)
# filename3 = '0.007_SP_elwoRe_UMa_lr1e-04_[288, 3, 256, 1024, 290]_ep14.pt'             ###
# checkpoint3 = torch.load('./simulation/result/model/'+filename3)         #
# filename4 = '0.008_SP_elwRe_UMa_lr1e-04_[288, 3, 256, 1024, 290]_ep17.pt'             ###
# checkpoint4 = torch.load('./simulation/result/model/'+filename4)         #

logits_net1 = checkpoint1['logits_net'].to(device)
# logits_net2 = checkpoint2['logits_net'].to(device)
# logits_net3 = checkpoint3['logits_net'].to(device) 
# logits_net4 = checkpoint4['logits_net'].to(device)                         #
# n_R, n_T, T = [checkpoint1['n_R'], checkpoint1['n_T'], checkpoint1['T']]
H_mean, H_sigma, W_mean = [0, 1, 0]

sqrt2 = 2**.5
SNR_dB = torch.arange(-4,10.1,2)
SNR_lin = 10**(SNR_dB/10.0)
NMSE_1 = torch.zeros_like(SNR_lin) # MLP 1
# NMSE_2 = torch.zeros_like(SNR_lin) # MLP 2
# NMSE_3 = torch.zeros_like(SNR_lin)
# NMSE_4 = torch.zeros_like(SNR_lin)
NMSE2 = torch.zeros_like(SNR_lin) # LS
NMSE4 = torch.zeros_like(SNR_lin) # sampled LMMSE

datasize_per_SNR = int(3e3)
test_data_size = datasize_per_SNR*len(SNR_dB)
n_R = checkpoint1['n_R']
n_T = checkpoint1['n_T']
T = checkpoint1['T']
n_I = checkpoint1['n_I']


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

pbar = tqdm(total = len(SNR_dB))
# def get_1snr_data(h_test, y_test, datasize_per_SNR, idx):
#     y_1snr = y_test[idx*datasize_per_SNR:(idx+1)*datasize_per_SNR, :]
#     h_1snr = h_test[idx*datasize_per_SNR:(idx+1)*datasize_per_SNR, :]
#     return h_1snr, y_1snr

for idx, snr in enumerate(SNR_lin):
    snr = snr.unsqueeze(0).to(device)
    # Pn = Pn.repeat(data_size).repeat_interleave(2).reshape(data_size, n_R*T, 2)
    # w = torch.view_as_complex(torch.normal(W_mean, torch.sqrt(Pn))/(sqrt2)).to(device) 
    # y = s + w

    D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

    IRS_coef_type = checkpoint1['IRS_coe_type']
    h_test, y_test, _, _ = importData(datasize_per_SNR, n_R, n_I, n_T, T, snr, device, W_Mean=0, IRScoef=IRS_coef_type)
    Psi = get_IRS_coef(IRS_coef_type,n_R,n_I,n_T,T).to(device)
    # h_1snr, y_1snr = get_1snr_data(h_test, y_test, datasize_per_SNR, idx)

    # Y_nmlz = (y.reshape(data_size,n_R,T)-h_mean2)/h_std2
    logits1 = logits_net1(y_test)
    test_tbh_cplx = turnCplx(logits1)
    norm_1 = torch.norm(turnCplx(h_test) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

    ''''''
    ### LS
    norm2 = torch.norm((turnCplx(h_test) - turnCplx(y_test)), dim=1)**2
    # norm2 = torch.norm((turnCplx(h_test) - torch.matmul(tbPsi.T, turnCplx(y_test).unsqueeze(2)).squeeze(2)/tbPsi.shape[0]), dim=1)**2

    ### LMMSE
    tbPsi = batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64)
    # if idx == 0:
    #     print(tbPsi.shape)
    #     print(h_test.shape)
    Sgnl = torch.matmul(tbPsi, turnCplx(h_test).unsqueeze(2)).squeeze(2)
    Ps = (Sgnl.abs()**2).mean(dim=0)
    Pn = Ps / snr
    cov_n = torch.diag(Pn)
    lmmse = LMMSE_solver(turnCplx(y_test).T, tbPsi, turnCplx(h_test).T, cov_n, datasize_per_SNR).T
    norm4 = torch.norm((turnCplx(h_test) - lmmse), dim=1)**2

    # logits2 = logits_net2(turnReal((y-1*h_mean2)/h_std2))
    # test_tbh_cplx = torch.view_as_complex(logits2.reshape(data_size, n_T*n_R+1, 2))
    # norm_2 = torch.norm(h - D.matmul(test_tbh_cplx.T).T*h_std2 - h_mean2, dim=1)**2
    # logits3 = logits_net3(torch.stack([Y_nmlz.real,Y_nmlz.imag,Y_nmlz.abs()],dim=1))
    # test_tbh_cplx = turnCplx(logits3)
    # norm_3 = torch.norm(h - D.matmul(test_tbh_cplx.T).T*h_std2-h_mean2, dim=1)**2
    # logits4 = logits_net4(torch.stack([Y_nmlz.real,Y_nmlz.imag,Y_nmlz.abs()],dim=1))
    # test_tbh_cplx = turnCplx(logits4)
    # norm_4 = torch.norm(h - D.matmul(test_tbh_cplx.T).T*h_std2-h_mean2, dim=1)**2


    with torch.no_grad():
        NMSE_1[idx] = 10*torch.log10((norm_1 / torch.norm(h_test, dim=1)**2).mean())
        # NMSE_2[idx] = 10*torch.log10((norm_2 / torch.norm((h), dim=1)**2).mean())
        # NMSE_3[idx] = 10*torch.log10((norm_3 / torch.norm((h), dim=1)**2).mean()) #
        # NMSE_4[idx] = 10*torch.log10((norm_4 / torch.norm((h), dim=1)**2).mean()) #

    NMSE2[idx] = 10*torch.log10((norm2 / torch.norm(h_test, dim=1)**2).mean())
    NMSE4[idx] = 10*torch.log10((norm4 / torch.norm(h_test, dim=1)**2).mean())
    
    # plt.text(10*torch.log10(snr)-1,NMSE4[idx]-1, f'({10**(NMSE4[idx].item()/10):.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr)-1,NMSE_2[idx]-1, f'({10**(NMSE_2[idx].item()/10):.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr),NMSE_3[idx], f'({10**(NMSE_3[idx].item()/10):.2f})')  ## plot linear NMSE value 
    pbar.update(1)
    
    

plt.plot(SNR_dB, NMSE2,'-o', label='LS', linewidth=1)
# plt.plot(SNR_dB, NMSE3,'-o', label='ideal LMMSE')
plt.plot(SNR_dB, NMSE4,'-o', label='LMMSE', linewidth=1 ,color="tab:red")
# plt.plot(SNR_dB, NMSE5,'-x', label='model-based PD LS', linewidth=1 ,color="tab:red")
# plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')

plt.plot(SNR_dB, NMSE_1,'-x', label='NP PD MMSE MIMO ChEst')   ###
# plt.plot(SNR_dB, NMSE_2,'-x', label='2 hidden layers mlp (3fc)')  ###
# plt.plot(SNR_dB, NMSE_3,'-x', label='channelNet w/o act. func.')  ###
# plt.plot(SNR_dB, NMSE_4,'-x', label='channelNet w/ act. func.')  ###

plt.suptitle("nonparametric PD trained MMSE MIMO ChEst with IRS vs SNR in Rayleigh")
# plt.title(' $[n_R,n_T]$:[4,36], MLP size:[288, 2048, 2048, 576] ')

# plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title('size:%s' %([2*n_R*T]+checkpoint1['hidden_sizes']+[2*n_R*n_T]))
plt.title('$[n_R,n_I,n_T,T]:[%s,%s,%s,%s], datasize: %s$' %(n_R,n_I,n_T,T,datasize_per_SNR))
plt.xlabel('SNR(dB)')
plt.ylabel('NMSE(dB)')
plt.legend()
plt.grid(True)
# plt.savefig('./simulation/result/snr/LS_mlp_Chest_-4_vs_i2.png')   ###
save_path = os.path.join(script_dir, 'snr', 'SP_ray_%s_vs_LS_LMMSE.pdf' %(IRS_coef_type))
plt.savefig(save_path)   ###

