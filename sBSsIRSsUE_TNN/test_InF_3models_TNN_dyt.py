import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# import LMMSE
from utils.batch_kronecker import batch_kronecker
from utils.IRS_ct_channels_dbm import importData
from utils.complex_utils import turnReal, turnCplx, vec
from utils.get_IRS_coef import get_IRS_coef
from utils.LMMSE import LMMSE_solver
# from utils.NN_model.MLP_2layer_1024 import MLP
from utils.NN_model.ISTANet import ISTANet
import os
import utils.IRS_ct_channels_TNN as IRS_ct_channels_TNN

torch.manual_seed(1)

device = torch.device("cuda:0")
# device1 = torch.device("cuda:1")
script_dir = os.path.dirname(os.path.abspath(__file__))
n_T = 4
n_I = 8
n_R = 4
T = 32

# MLPs
filename1 = os.path.join(script_dir, 'trained_model', '4.059_SP_inf_MLP_psi_i_lr1e-03_[256, 1024, 258]_ep51.pt')
checkpoint1 = torch.load(filename1, weights_only=False, map_location=device)
filename2 = os.path.join(script_dir, 'trained_model', '1.580_SP_inf_MLP_psi_d_lr1e-03_[256, 1024, 258]_ep65.pt')
checkpoint2 = torch.load(filename2, weights_only=False, map_location=device)
filename3 = os.path.join(script_dir, 'trained_model', '1.708_SP_inf_MLP_psi_h_lr1e-03_[256, 1024, 258]_ep85.pt')
checkpoint3 = torch.load(filename3, weights_only=False, map_location=device)

filename11 = os.path.join(script_dir, 'trained_model', '2.479_SP_inf_MLP_dyt_psi_d_lr1e-04_[256, 1024, 258]_ep36.pt')
checkpoint11 = torch.load(filename11, weights_only=False, map_location=device)

# # channelNets
# filename4 = os.path.join(script_dir, 'trained_model/InF_trained_model', '1.206_elbir_psi_i_lr1e-04_ep6.pt')
# checkpoint4 = torch.load(filename4, weights_only=False, map_location=device)
# filename5 = os.path.join(script_dir, 'trained_model/InF_trained_model', '1.275_elbir_psi_d_lr1e-04_ep6.pt')
# checkpoint5 = torch.load(filename5, weights_only=False, map_location=device)
# filename6 = os.path.join(script_dir, 'trained_model/InF_trained_model', '1.406_elbir_psi_h_lr1e-04_ep6.pt')
# checkpoint6 = torch.load(filename6, weights_only=False, map_location=device)

# # ISTA-Nets
# filename7 = os.path.join(script_dir, 'trained_model/InF_trained_model', '0.789_inf_ISTA_psi_i_lr1e-03_ep28.pt')
# checkpoint7 = torch.load(filename7, weights_only=False, map_location=device1)
# filename8 = os.path.join(script_dir, 'trained_model/InF_trained_model', '0.788_inf_ISTA_psi_d_lr1e-03_ep37.pt')
# checkpoint8 = torch.load(filename8, weights_only=False, map_location=device1)
# filename9 = os.path.join(script_dir, 'trained_model/InF_trained_model', '0.907_inf_ISTA_psi_h_lr1e-03_ep29.pt')
# checkpoint9 = torch.load(filename9, weights_only=False, map_location=device1)
	
filename10 = os.path.join(script_dir, 'trained_model', '1.184_SP_inf_MLP_TNN_lr1e-03_[256, 1024, 258]_ep74.pt')
checkpoint10 = torch.load(filename10, weights_only=False, map_location=device)
Psi = checkpoint10['Psi'].to(device)
# print(Psi)

logits_net1 = checkpoint1['logits_net'].to(device)
logits_net2 = checkpoint2['logits_net'].to(device)
logits_net3 = checkpoint3['logits_net'].to(device) 
logits_net11 = checkpoint11['logits_net'].to(device) 
logits_net10 = checkpoint10['logits_net'].to(device)
tnn = checkpoint10['tnn'].to(device)


# logits_net4 = checkpoint4['logits_net'].to(device)
# logits_net5 = checkpoint5['logits_net'].to(device)
# logits_net6 = checkpoint6['logits_net'].to(device) 

# logits_net7 = checkpoint7['logits_net'].to('cuda:0')
# logits_net8 = checkpoint8['logits_net'].to('cuda:1') 
# logits_net9 = checkpoint9['logits_net'].to('cuda:2')

sqrt2 = 2**.5
SNR_dB = torch.arange(-4,10.1,2)
SNR_lin = 10**(SNR_dB/10.0).to(device)
NMSE_1 = torch.zeros_like(SNR_lin) # MLP 1
NMSE_2 = torch.zeros_like(SNR_lin) 
NMSE_3 = torch.zeros_like(SNR_lin)
NMSE_4 = torch.zeros_like(SNR_lin) # channelNet
NMSE_5 = torch.zeros_like(SNR_lin) 
NMSE_6 = torch.zeros_like(SNR_lin)
NMSE_7 = torch.zeros_like(SNR_lin) # ISTA-Net 
NMSE_8 = torch.zeros_like(SNR_lin) 
NMSE_9 = torch.zeros_like(SNR_lin)
NMSE_10 = torch.zeros_like(SNR_lin)
NMSE_11 = torch.zeros_like(SNR_lin)

NMSE_LS_i = torch.zeros_like(SNR_lin) # LS
NMSE_LM_i = torch.zeros_like(SNR_lin) # sampled LMMSE
NMSE_LS_d = torch.zeros_like(SNR_lin) # LS
NMSE_LM_d = torch.zeros_like(SNR_lin) # sampled LMMSE
NMSE_LS_h = torch.zeros_like(SNR_lin) # LS
NMSE_LM_h = torch.zeros_like(SNR_lin) # sampled LMMSE

test_size = int(2.4e4)
datasize_per_SNR = test_size//len(SNR_lin)
test_data_size = datasize_per_SNR*len(SNR_dB)

### import testing data ###
channel = 'InF'
h_test, y_test_i, h_mean, h_std = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='i', phase = 'test', channel=channel)
_, y_test_d, _, _ = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='d', phase = 'test', channel=channel)
_, y_test_h, _, _ = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='h', phase = 'test', channel=channel)
h_test = h_test.reshape(len(SNR_lin), datasize_per_SNR, n_R*n_T*n_I*2)
y_test_i = y_test_i.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
y_test_h = y_test_h.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
y_test_d = y_test_d.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
# Y_test_nmlz_i = (torch.view_as_complex(y_test_i.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std
# Y_test_nmlz_d = (torch.view_as_complex(y_test_d.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std
# Y_test_nmlz_h = (torch.view_as_complex(y_test_h.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std
_, y_test_TNN, _, _ = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=Psi, phase = 'test', channel=channel)
y_test_TNN = y_test_TNN.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
# y_test = add_noise(tnn(turnCplx(h_test)), SNR_lin, test_size, n_R, T, device)
# logits_test = logits_net(turnReal(y_test - h_mean)/h_std)


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

D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

pbar = tqdm(total = len(SNR_dB))
for idx, snr in enumerate(SNR_lin):
    snr = snr.unsqueeze(0).to(device)


    # IRS_coef_type = checkpoint1['IRS_coe_type']

    with torch.no_grad():
        # Y_nmlz = (y.reshape(data_size,n_R,T)-h_mean2)/h_std2
        logits1 = logits_net1(turnReal(turnCplx(y_test_i[idx])-h_mean)/h_std)
        test_tbh_cplx = turnCplx(logits1)*h_std + h_mean
        norm_1 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        logits2 = logits_net2(turnReal(turnCplx(y_test_d[idx])-h_mean)/h_std)
        test_tbh_cplx = turnCplx(logits2)*h_std + h_mean
        norm_2 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        logits3 = logits_net3(turnReal(turnCplx(y_test_h[idx])-h_mean)/h_std)
        test_tbh_cplx = turnCplx(logits3)*h_std + h_mean
        norm_3 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2
        
        logits11 = logits_net11(y_test_h[idx])
        test_tbh_cplx = turnCplx(logits11)
        norm_11 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2
	
        # y_test_TNN = IRS_ct_channels_TNN.add_noise(tnn(turnCplx(h_test[idx])), snr, y_test_h[idx].shape[0], n_R, T, device)
        # logits10 = logits_net10(turnReal(y_test_TNN - h_mean)/h_std)
        # test_tbh_cplx = turnCplx(logits10)*h_std + h_mean
        # norm_10 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2
        # y_test_TNN = IRS_ct_channels_TNN.add_noise(tnn(turnCplx(h_test[idx])), snr, y_test_h[idx].shape[0], n_R, T, device)
        logits10 = logits_net10(turnReal(turnCplx(y_test_TNN[idx]) - h_mean)/h_std)
        test_tbh_cplx = turnCplx(logits10)*h_std + h_mean
        norm_10 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

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

        # ### ISTA-Net
        # # torch.cuda.empty_cache()
        # logits7, _ = logits_net7((turnReal(turnCplx(y_test_i[idx])-h_mean)/h_std))
        # test_tbh_cplx = turnCplx(logits7).to(device)*h_std + h_mean
        # norm_7 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits8, _ = logits_net8((turnReal(turnCplx(y_test_d[idx])-h_mean)/h_std))
        # test_tbh_cplx = turnCplx(logits8).to(device)*h_std + h_mean
        # norm_8 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits9, _ = logits_net9((turnReal(turnCplx(y_test_h[idx])-h_mean)/h_std))
        # test_tbh_cplx = turnCplx(logits9).to(device)*h_std + h_mean
        # norm_9 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2

        '''
        LS and LMMSE numerical solution for different IRS coefficient matrix
        '''

        norm_LS_i, norm_LM_i = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_i[idx], IRS_coef_type='i')
        norm_LS_d, norm_LM_d = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_d[idx], IRS_coef_type='d')
        norm_LS_h, norm_LM_h = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_h[idx], IRS_coef_type='h')


        NMSE_1[idx] = 10*torch.log10((norm_1 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_2[idx] = 10*torch.log10((norm_2 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_3[idx] = 10*torch.log10((norm_3 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_4[idx] = 10*torch.log10((norm_4 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_5[idx] = 10*torch.log10((norm_5 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_6[idx] = 10*torch.log10((norm_6 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_7[idx] = 10*torch.log10((norm_7 / torch.norm(h_test[idx], dim=1)**2).mean())
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

SNR_dB = torch.tensor([-44.1029, -46.1028, -48.1029, -50.1028, -52.1029, -54.1029, -56.1028, -58.1029])


plt.plot(SNR_dB.flip(0), NMSE_LS_i.to('cpu'), label='LS w/ I', linewidth=1, linestyle='--', marker='o', color="tab:blue")
plt.plot(SNR_dB.flip(0), NMSE_LS_d.to('cpu'), label='LS w/ D', linewidth=1, linestyle=':', marker='o', color="tab:blue")
plt.plot(SNR_dB.flip(0), NMSE_LS_h.to('cpu'), label='LS w/ H', linewidth=1, linestyle='-', marker='o', color="tab:blue")
plt.plot(SNR_dB.flip(0), NMSE_LM_i.to('cpu'), label='LMMSE w/ I', linewidth=1, linestyle='--', marker='o', color="tab:red")
plt.plot(SNR_dB.flip(0), NMSE_LM_d.to('cpu'), label='LMMSE w/ D', linewidth=1, linestyle=':', marker='o', color="tab:red")
plt.plot(SNR_dB.flip(0), NMSE_LM_h.to('cpu'), label='LMMSE w/ H', linewidth=1, linestyle='-', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE5,'-x', label='model-based PD LS', linewidth=1 ,color="tab:red")
# plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')

# plt.plot(SNR_dB, NMSE_4.to('cpu'), label='channelNet w/ I ', linewidth=1, linestyle='--', marker='x', color="tab:green")  ###
# plt.plot(SNR_dB, NMSE_5.to('cpu'), label='channelNet w/ D ', linewidth=1, linestyle=':', marker='x', color="tab:green")  ###
# plt.plot(SNR_dB, NMSE_6.to('cpu'), label='channelNet w/ H ', linewidth=1, linestyle='-', marker='x', color="tab:green")  ###

# plt.plot(SNR_dB, NMSE_7.to('cpu'), label='ISTANet w/ I ', linewidth=1, linestyle='--', marker='x', color="tab:brown")  ###
# plt.plot(SNR_dB, NMSE_8.to('cpu'), label='ISTANet w/ D ', linewidth=1, linestyle=':', marker='x', color="tab:brown")  ###
# plt.plot(SNR_dB, NMSE_9.to('cpu'), label='ISTANet w/ H ', linewidth=1, linestyle='-', marker='x', color="tab:brown")  ###

plt.plot(SNR_dB.flip(0), NMSE_1.to('cpu'), label='NP PD w/ I', linewidth=1, linestyle='--', marker='x', color="tab:orange")   ###
plt.plot(SNR_dB.flip(0), NMSE_2.to('cpu'), label='NP PD w/ D', linewidth=1, linestyle=':', marker='x', color="tab:orange")  ###
plt.plot(SNR_dB.flip(0), NMSE_3.to('cpu'), label='NP PD w/ H', linewidth=1, linestyle='-', marker='x', color="tab:orange")  ###
plt.plot(SNR_dB.flip(0), NMSE_11.to('cpu'), label='NP PD w/ D dyt', linewidth=1, linestyle='-', marker='x', color="black")  ###

plt.plot(SNR_dB.flip(0), NMSE_10.to('cpu'), label='NP PD w/ TNN', linewidth=1, linestyle='-', marker='x', color="tab:brown")  ###

# plt.plot(SNR_dB, NMSE_3,'-x', label='channelNet w/o act. func.')  ###
# plt.plot(SNR_dB, NMSE_4,'-x', label='channelNet w/ act. func.')  ###

plt.suptitle("NP PD trained MMSE MIMO nmlz ChEst with IRS vs SNR in InF 2.5GHz")
# plt.title(' $[n_R,n_T]$:[4,36], MLP size:[288, 2048, 2048, 576] ')

# plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title('size:%s' %([2*n_R*T]+checkpoint1['hidden_sizes']+[2*n_R*n_T]))
plt.title('$[n_R,n_I,n_T,T]:[%s,%s,%s,%s], datasize: %s$' %(n_R,n_I,n_T,T,test_size))
plt.xlabel('SNR(dB)')
plt.ylabel('NMSE(dB)')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize='small')
plt.grid(True)
# plt.savefig('./simulation/result/snr/LS_mlp_Chest_-4_vs_i2.png')   ###
save_path = os.path.join(script_dir, 'ChEsts_testing_performance', 'SP_%s_MLP_vs_CN_IN_TNN_dbm_dyt.png'%(channel)) #  %(IRS_coef_type)
plt.savefig(save_path)   ###

