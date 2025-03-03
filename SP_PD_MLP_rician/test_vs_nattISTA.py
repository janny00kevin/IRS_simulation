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
from utils.NN_model.ISTANet import ISTANet
import os

torch.manual_seed(0)

device = torch.device("cuda:0")
device1 = torch.device("cuda:1")
script_dir = os.path.dirname(os.path.abspath(__file__))
n_R = 4
n_T = 4
T = 32
n_I = 8

# # MLPs
# filename1 = os.path.join(script_dir, 'result', '0.160_SP_ray_MLP_psi_i_lr1e-04_[256, 1024, 258]_ep300.pt')
# checkpoint1 = torch.load(filename1, weights_only=False)
# filename2 = os.path.join(script_dir, 'result', '0.160_SP_ray_MLP_psi_d_lr1e-04_[256, 1024, 258]_ep150.pt')
# checkpoint2 = torch.load(filename2, weights_only=False)
# filename3 = os.path.join(script_dir, 'result', '0.160_SP_ray_MLP_psi_h_lr1e-04_[256, 1024, 258]_ep150.pt')
# checkpoint3 = torch.load(filename3, weights_only=False)

# # channelNets
# filename4 = os.path.join(script_dir, 'result', '0.207_SP_ric_elbir_psi_i_lr1e-02_ep8.pt')
# checkpoint4 = torch.load(filename4, weights_only=False, map_location=device)
# filename5 = os.path.join(script_dir, 'result', '0.197_SP_ric_elbir_psi_d_lr1e-02_ep11.pt')
# checkpoint5 = torch.load(filename5, weights_only=False, map_location=device)
# filename6 = os.path.join(script_dir, 'result', '0.214_SP_ric_elbir_psi_h_lr1e-02_ep8.pt')
# checkpoint6 = torch.load(filename6, weights_only=False, map_location=device)

# ISTA-Nets
filename7 = os.path.join(script_dir, 'result', '0.155_SP_ric_ISTA_psi_i_lr1e-03_ep17.pt')
checkpoint7 = torch.load(filename7, weights_only=False, map_location=device1)
# filename8 = os.path.join(script_dir, 'result', '0.156_SP_ric_ISTA_psi_d_lr1e-03_ep41.pt')
# checkpoint8 = torch.load(filename8, weights_only=False, map_location=device1)
# filename9 = os.path.join(script_dir, 'result', '0.157_SP_ric_ISTA_psi_h_lr1e-03_ep37.pt')
# checkpoint9 = torch.load(filename9, weights_only=False, map_location=device1)

filename10 = os.path.join(script_dir, 'result', '0.000_ric_nattISTA_psi_i_ep30.pt')
checkpoint10 = torch.load(filename10, weights_only=False)

filename11 = os.path.join(script_dir, 'result', '0.012_ric_nattISTA_nmlz_psi_i_ep55.pt')
checkpoint11 = torch.load(filename11, weights_only=False)

# h_mean2 = checkpoint2['h_mean'].to(device)
# h_std2 = checkpoint2['h_std'].to(device)
# filename3 = '0.007_SP_elwoRe_UMa_lr1e-04_[288, 3, 256, 1024, 290]_ep14.pt'             ###
# checkpoint3 = torch.load('./simulation/result/model/'+filename3)         #
# filename4 = '0.008_SP_elwRe_UMa_lr1e-04_[288, 3, 256, 1024, 290]_ep17.pt'             ###
# checkpoint4 = torch.load('./simulation/result/model/'+filename4)         #
# def move_model_to_device(model, device):
#     model.to(device)
#     for param in model.parameters():
#         param.data = param.data.to(device)
#     return model

# logits_net1 = checkpoint1['logits_net'].to(device)
# logits_net2 = checkpoint2['logits_net'].to(device)
# logits_net3 = checkpoint3['logits_net'].to(device) 

# logits_net4 = checkpoint4['logits_net'].to(device)
# logits_net5 = checkpoint5['logits_net'].to(device)
# logits_net6 = checkpoint6['logits_net'].to(device) 

logits_net7 = checkpoint7['logits_net'].to('cuda:0')
# logits_net8 = checkpoint8['logits_net'].to('cuda:1') 
# logits_net9 = checkpoint9['logits_net'].to('cuda:2')

logits_net10 = checkpoint10['logits_net'].to('cuda:0')
logits_net11 = checkpoint11['logits_net'].to('cuda:2')

# logits_net7 = ISTANet(5, device1, n_R, n_I, n_T, T)
# logits_net8 = ISTANet(5, device1, n_R, n_I, n_T, T)
# logits_net9 = ISTANet(5, device1, n_R, n_I, n_T, T)

# logits_net7.load_state_dict(checkpoint7['logits_net'])
# logits_net8.load_state_dict(checkpoint8['logits_net'])
# logits_net9.load_state_dict(checkpoint9['logits_net'])

# logits_net7 = logits_net7.to(device1)
# logits_net8 = logits_net8.to(device1)
# logits_net9 = logits_net9.to(device1)


sqrt2 = 2**.5
SNR_dB = torch.arange(-4,10.1,2)
SNR_lin = 10**(SNR_dB/10.0).to(device)
# NMSE_1 = torch.zeros_like(SNR_lin) # MLP 1
# NMSE_2 = torch.zeros_like(SNR_lin) # MLP 2
# NMSE_3 = torch.zeros_like(SNR_lin)
# NMSE_4 = torch.zeros_like(SNR_lin) # channelNet
# NMSE_5 = torch.zeros_like(SNR_lin) 
# NMSE_6 = torch.zeros_like(SNR_lin)
NMSE_7 = torch.zeros_like(SNR_lin) # ISTA-Net 
# NMSE_8 = torch.zeros_like(SNR_lin) 
# NMSE_9 = torch.zeros_like(SNR_lin)
NMSE_10 = torch.zeros_like(SNR_lin) # Natt ISTA-Net W/O normalization
NMSE_11 = torch.zeros_like(SNR_lin) # Natt ISTA-Net W/ normalization

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
Y_test_nmlz_i = (torch.view_as_complex(y_test_i.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std
Y_test_nmlz_d = (torch.view_as_complex(y_test_d.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std
Y_test_nmlz_h = (torch.view_as_complex(y_test_h.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std


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
    LS_solution = torch.matmul(tbPsi.pinverse(), turnCplx(y_test).unsqueeze(2)).squeeze(2)
    # Compute the norm of the difference
    norm_LS = torch.norm((turnCplx(h_test) - LS_solution), dim=1)**2

    ### LMMSE
    tbPsi = batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64)
    Sgnl = torch.matmul(tbPsi, turnCplx(h_test).unsqueeze(2)).squeeze(2)
    Ps = (Sgnl.abs()**2).mean(dim=0)
    Pn = Ps / snr
    cov_n = torch.diag(Pn)
    lmmse = LMMSE_solver(turnCplx(y_test).T, tbPsi, turnCplx(h_test).T, cov_n, datasize_per_SNR).T
    norm_LM = torch.norm((turnCplx(h_test) - lmmse), dim=1)**2
    return norm_LS,norm_LM

pbar = tqdm(total = len(SNR_dB))
for idx, snr in enumerate(SNR_lin):
    snr = snr.unsqueeze(0).to(device)

    # D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

    # IRS_coef_type = checkpoint1['IRS_coe_type']

    with torch.no_grad():
        # # Y_nmlz = (y.reshape(data_size,n_R,T)-h_mean2)/h_std2
        # logits1 = logits_net1(turnReal(turnCplx(y_test_i[idx])-h_mean)/h_std)
        # test_tbh_cplx = turnCplx(logits1)*h_std + h_mean
        # norm_1 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # logits2 = logits_net2(turnReal(turnCplx(y_test_d[idx])-h_mean)/h_std)
        # test_tbh_cplx = turnCplx(logits2)*h_std + h_mean
        # norm_2 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # logits3 = logits_net3(turnReal(turnCplx(y_test_h[idx])-h_mean)/h_std)
        # test_tbh_cplx = turnCplx(logits3)*h_std + h_mean
        # norm_3 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # ### channelNet
        # logits4 = logits_net4(torch.stack([Y_test_nmlz_i[idx].real,Y_test_nmlz_i[idx].imag,Y_test_nmlz_i[idx].abs()],dim=1))
        # test_tbh_cplx = turnCplx(logits4)*h_std + h_mean
        # norm_4 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits5 = logits_net5(torch.stack([Y_test_nmlz_d[idx].real,Y_test_nmlz_d[idx].imag,Y_test_nmlz_d[idx].abs()],dim=1))
        # test_tbh_cplx = turnCplx(logits5)*h_std + h_mean
        # norm_5 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits6 = logits_net6(torch.stack([Y_test_nmlz_h[idx].real,Y_test_nmlz_h[idx].imag,Y_test_nmlz_h[idx].abs()],dim=1))
        # test_tbh_cplx = turnCplx(logits6)*h_std + h_mean
        # norm_6 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2

        ### ISTA-Net
        # torch.cuda.empty_cache()
        logits7, _ = logits_net7((turnReal(turnCplx(y_test_i[idx])-h_mean)/h_std))
        test_tbh_cplx = turnCplx(logits7).to(device)*h_std + h_mean
        norm_7 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # [logits8, error] = logits_net8((turnReal(turnCplx(y_test_d[idx])-h_mean)/h_std))
        # test_tbh_cplx = turnCplx(logits8).to(device)*h_std + h_mean
        # norm_8 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # [logits9, error] = logits_net9((turnReal(turnCplx(y_test_h[idx])-h_mean)/h_std))
        # test_tbh_cplx = turnCplx(logits9).to(device)*h_std + h_mean
        # norm_9 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2

        ### Natt's ISTA-Net W/O normalization
        logits10, _ = logits_net10(y_test_i[idx].to(torch.float32))
        norm_10 = torch.norm(h_test[idx].to(torch.float32) - logits10, dim=1)**2

        ### Natt's ISTA-Net W/ normalization
        logits11, _ = logits_net11((turnReal(turnCplx(y_test_i[idx])-h_mean)/h_std))
        test_tbh_cplx = turnCplx(logits11).to(device)*h_std + h_mean
        norm_11 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2

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


        # NMSE_1[idx] = 10*torch.log10((norm_1 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_2[idx] = 10*torch.log10((norm_2 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_3[idx] = 10*torch.log10((norm_3 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_4[idx] = 10*torch.log10((norm_4 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_5[idx] = 10*torch.log10((norm_5 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_6[idx] = 10*torch.log10((norm_6 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_7[idx] = 10*torch.log10((norm_7 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_8[idx] = 10*torch.log10((norm_8 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_9[idx] = 10*torch.log10((norm_9 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_10[idx] = 10*torch.log10((norm_10 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_11[idx] = 10*torch.log10((norm_11 / torch.norm(h_test[idx], dim=1)**2).mean())

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

    

plt.plot(SNR_dB, NMSE_LS_i.to('cpu'), label='LS w/ I', linewidth=1, linestyle='-', marker='o', color="tab:blue")
# plt.plot(SNR_dB, NMSE_LS_d.to('cpu'), label='LS w/ D', linewidth=1, linestyle=':', marker='o', color="tab:blue")
# plt.plot(SNR_dB, NMSE_LS_h.to('cpu'), label='LS w/ H', linewidth=1, linestyle='-', marker='o', color="tab:blue")
plt.plot(SNR_dB, NMSE_LM_i.to('cpu'), label='LMMSE w/ I', linewidth=1, linestyle='-', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE_LM_d.to('cpu'), label='LMMSE w/ D', linewidth=1, linestyle=':', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE_LM_h.to('cpu'), label='LMMSE w/ H', linewidth=1, linestyle='-', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE5,'-x', label='model-based PD LS', linewidth=1 ,color="tab:red")
# plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')

# plt.plot(SNR_dB, NMSE_4.to('cpu'), label='channelNet w/ I ', linewidth=1, linestyle='--', marker='x', color="tab:green")  ###
# plt.plot(SNR_dB, NMSE_5.to('cpu'), label='channelNet w/ D ', linewidth=1, linestyle=':', marker='x', color="tab:green")  ###
# plt.plot(SNR_dB, NMSE_6.to('cpu'), label='channelNet w/ H ', linewidth=1, linestyle='-', marker='x', color="tab:green")  ###

plt.plot(SNR_dB, NMSE_7.to('cpu'), label='My ISTANet w/ I ', linewidth=1, linestyle='--', marker='x', color="tab:cyan")  ###
# plt.plot(SNR_dB, NMSE_8.to('cpu'), label='ISTANet w/ D ', linewidth=1, linestyle=':', marker='x', color="tab:brown")  ###
# plt.plot(SNR_dB, NMSE_9.to('cpu'), label='ISTANet w/ H ', linewidth=1, linestyle='-', marker='x', color="tab:brown")  ###

# plt.plot(SNR_dB, NMSE_1.to('cpu'), label='NP PD w/ I', linewidth=1, linestyle='--', marker='x', color="tab:orange")   ###
# plt.plot(SNR_dB, NMSE_2.to('cpu'), label='NP PD w/ D', linewidth=1, linestyle=':', marker='x', color="tab:orange")  ###
# plt.plot(SNR_dB, NMSE_3.to('cpu'), label='NP PD w/ H', linewidth=1, linestyle='-', marker='x', color="tab:orange")  ###

plt.plot(SNR_dB, NMSE_10.to('cpu'), label='ISTANet w/o nmlz w/ I ', linewidth=1, linestyle='--', marker='x', color="tab:green")  ###
plt.plot(SNR_dB, NMSE_11.to('cpu'), label='ISTANet w/  nmlz w/ I ', linewidth=1, linestyle=':', marker='x', color="tab:green")  ###

# plt.plot(SNR_dB, NMSE_3,'-x', label='channelNet w/o act. func.')  ###
# plt.plot(SNR_dB, NMSE_4,'-x', label='channelNet w/ act. func.')  ###

plt.suptitle("ISTA-Net ChEst with IRS vs SNR in Rician fading channel")
# plt.title(' $[n_R,n_T]$:[4,36], MLP size:[288, 2048, 2048, 576] ')

# plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title('size:%s' %([2*n_R*T]+checkpoint1['hidden_sizes']+[2*n_R*n_T]))
plt.title('$[n_R,n_I,n_T,T]:[%s,%s,%s,%s], datasize: %s$' %(n_R,n_I,n_T,T,datasize_per_SNR))
plt.xlabel('SNR(dB)')
plt.ylabel('NMSE(dB)')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
plt.grid(True)
# plt.savefig('./simulation/result/snr/LS_mlp_Chest_-4_vs_i2.png')   ###
save_path = os.path.join(script_dir, 'snr', 'SP_ric_IN_vs_Natt_vs_nmlz.pdf') #  %(IRS_coef_type)
plt.savefig(save_path)   ###

