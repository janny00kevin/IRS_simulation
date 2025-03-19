import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# import LMMSE
from utils.batch_kronecker import batch_kronecker
from utils.IRS_ct_channels_align import importData
from utils.complex_utils import turnReal, turnCplx, vec
from utils.get_IRS_coef import get_IRS_coef
from utils.LMMSE import LMMSE_solver
import os

torch.manual_seed(0)

device = torch.device("cuda:0")
device1 = torch.device("cuda:1")
script_dir = os.path.dirname(os.path.abspath(__file__))
n_T = 4
n_I = 8
n_R = 1
T = 32

# MLPs
filename1 = os.path.join(script_dir, 'trained_model', '1.310_PD_uma_MLP_psi_ho_lr1e-03_[64, 1024, 66]_ep78.pt')
checkpoint1 = torch.load(filename1, weights_only=False)
filename2 = os.path.join(script_dir, 'trained_model', '1.722_PD_uma_MLP_psi_ha_lr1e-03_[64, 1024, 66]_ep68.pt')
checkpoint2 = torch.load(filename2, weights_only=False)
# filename3 = os.path.join(script_dir, 'trained_model', '0.954_SP_uma_MLP_psi_h_lr1e-03_[256, 1024, 258]_ep103.pt')
# checkpoint3 = torch.load(filename3, weights_only=False)

# # channelNets
filename4 = os.path.join(script_dir, 'trained_model', '0.997_uma_elbir_psi_ho_lr1e-04_ep6.pt')
checkpoint4 = torch.load(filename4, weights_only=False, map_location=device)
filename5 = os.path.join(script_dir, 'trained_model', '0.997_uma_elbir_psi_ha_lr1e-04_ep6.pt')
checkpoint5 = torch.load(filename5, weights_only=False, map_location=device)
# filename6 = os.path.join(script_dir, 'trained_model', '0.652_SP_rt_elbir_psi_h_lr1e-04_ep20.pt')
# checkpoint6 = torch.load(filename6, weights_only=False, map_location=device)

# ISTA-Nets
filename7 = os.path.join(script_dir, 'trained_model', '1.008_uma_ISTA_psi_ho_lr1e-03_ep15.pt')
checkpoint7 = torch.load(filename7, weights_only=False, map_location=device1)
filename8 = os.path.join(script_dir, 'trained_model', '1.111_uma_ISTA_psi_ha_lr1e-03_ep31.pt')
checkpoint8 = torch.load(filename8, weights_only=False, map_location=device1)
# filename9 = os.path.join(script_dir, 'trained_model', '0.709_SP_uma_ISTA_psi_h_lr1e-03_ep19.pt')
# checkpoint9 = torch.load(filename9, weights_only=False, map_location=device1)


logits_net1 = checkpoint1['logits_net'].to(device)
logits_net2 = checkpoint2['logits_net'].to(device)
# logits_net3 = checkpoint3['logits_net'].to(device) 

logits_net4 = checkpoint4['logits_net'].to(device)
logits_net5 = checkpoint5['logits_net'].to(device)
# logits_net6 = checkpoint6['logits_net'].to(device) 

logits_net7 = checkpoint7['logits_net'].to('cuda:0')
logits_net8 = checkpoint8['logits_net'].to('cuda:1') 
# logits_net9 = checkpoint9['logits_net'].to('cuda:2')

sqrt2 = 2**.5
SNR_dB = torch.arange(-4,10.1,2)
SNR_lin = 10**(SNR_dB/10.0).to(device)
NMSE_1 = torch.zeros_like(SNR_lin) # MLP 1
NMSE_2 = torch.zeros_like(SNR_lin)
# NMSE_3 = torch.zeros_like(SNR_lin)
NMSE_4 = torch.zeros_like(SNR_lin) # channelNet
NMSE_5 = torch.zeros_like(SNR_lin) 
# NMSE_6 = torch.zeros_like(SNR_lin)
NMSE_7 = torch.zeros_like(SNR_lin) # ISTA-Net 
NMSE_8 = torch.zeros_like(SNR_lin) 
# NMSE_9 = torch.zeros_like(SNR_lin)

# NMSE_LS_i = torch.zeros_like(SNR_lin) # LS
# NMSE_LM_i = torch.zeros_like(SNR_lin) # sampled LMMSE
# NMSE_LS_d = torch.zeros_like(SNR_lin) # LS
# NMSE_LM_d = torch.zeros_like(SNR_lin) # sampled LMMSE
NMSE_LS_h_o = torch.zeros_like(SNR_lin) # LS
NMSE_LM_h_o = torch.zeros_like(SNR_lin) # sampled LMMSE
NMSE_LS_h_a = torch.zeros_like(SNR_lin) 
NMSE_LM_h_a = torch.zeros_like(SNR_lin) 

test_size = int(2.4e4)
datasize_per_SNR = test_size//len(SNR_lin)
test_data_size = datasize_per_SNR*len(SNR_dB)

### import testing data ###
channel = 'uma'
# h_test_o, y_test_i, h_mean_o, h_std_o = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='i', case = 'test', channel=channel, config = 'o')
# _, y_test_d, _, _ = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='d', phase = 'test', channel=channel, config = 'o')
h_test_o, y_test_h_o, h_mean_o, h_std_o = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='h', phase = 'test', channel=channel, config = 'o')
h_test_o = h_test_o.reshape(len(SNR_lin), datasize_per_SNR, n_R*n_T*n_I*2)
h_test_a, y_test_h_a, h_mean_a, h_std_a = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='h', phase = 'test', channel=channel, config = 'a')
h_test_a = h_test_a.reshape(len(SNR_lin), datasize_per_SNR, n_R*n_T*n_I*2)
# y_test_i = y_test_i.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
# y_test_d = y_test_d.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
y_test_h_o = y_test_h_o.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
y_test_h_a = y_test_h_a.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
# Y_test_nmlz_i = (torch.view_as_complex(y_test_i.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean_o)/h_std_o
# Y_test_nmlz_d = (torch.view_as_complex(y_test_d.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean_o)/h_std_o
Y_test_nmlz_h_o = (torch.view_as_complex(y_test_h_o.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean_o)/h_std_o
Y_test_nmlz_h_a = (torch.view_as_complex(y_test_h_a.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean_a)/h_std_a


def LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test_o, y_test, IRS_coef_type):
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
        h_test_o (torch.Tensor): The test channel matrix.
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
    norm_LS = torch.norm((turnCplx(h_test_o) - LS_solution), dim=1)**2

    ### LMMSE
    tbPsi = batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64)
    Sgnl = torch.matmul(tbPsi, turnCplx(h_test_o).unsqueeze(2)).squeeze(2)
    Ps = (Sgnl.abs()**2).mean(dim=0)
    Pn = Ps / snr
    cov_n = torch.diag(Pn)
    lmmse = LMMSE_solver(turnCplx(y_test).T, tbPsi, turnCplx(h_test_o).T, cov_n, datasize_per_SNR).T
    norm_LM = torch.norm((turnCplx(h_test_o) - lmmse), dim=1)**2
    return norm_LS,norm_LM

pbar = tqdm(total = len(SNR_dB))
for idx, snr in enumerate(SNR_lin):
    snr = snr.unsqueeze(0).to(device)

    D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

    # IRS_coef_type = checkpoint1['IRS_coe_type']

    with torch.no_grad():
        # Y_nmlz = (y.reshape(data_size,n_R,T)-h_mean2)/h_std2
        logits1 = logits_net1(turnReal(turnCplx(y_test_h_o[idx])-h_mean_o)/h_std_o)
        test_tbh_cplx = turnCplx(logits1)*h_std_o + h_mean_o
        norm_1 = torch.norm(turnCplx(h_test_o[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        logits2 = logits_net2(turnReal(turnCplx(y_test_h_a[idx])-h_mean_a)/h_std_a)
        test_tbh_cplx = turnCplx(logits2)*h_std_a + h_mean_a
        norm_2 = torch.norm(turnCplx(h_test_a[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # logits3 = logits_net3(turnReal(turnCplx(y_test_h_o[idx])-h_mean_o)/h_std_o)
        # test_tbh_cplx = turnCplx(logits3)*h_std_o + h_mean_o
        # norm_3 = torch.norm(turnCplx(h_test_o[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        ### channelNet
        logits4 = logits_net4(torch.stack([Y_test_nmlz_h_o[idx].real,Y_test_nmlz_h_o[idx].imag,Y_test_nmlz_h_o[idx].abs()],dim=1))
        test_tbh_cplx = turnCplx(logits4)*h_std_o + h_mean_o
        norm_4 = torch.norm(turnCplx(h_test_o[idx]) - test_tbh_cplx, dim=1)**2
        
        logits5 = logits_net5(torch.stack([Y_test_nmlz_h_a[idx].real,Y_test_nmlz_h_a[idx].imag,Y_test_nmlz_h_a[idx].abs()],dim=1))
        test_tbh_cplx = turnCplx(logits5)*h_std_a + h_mean_a
        norm_5 = torch.norm(turnCplx(h_test_a[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits6 = logits_net6(torch.stack([Y_test_nmlz_h[idx].real,Y_test_nmlz_h[idx].imag,Y_test_nmlz_h[idx].abs()],dim=1))
        # test_tbh_cplx = turnCplx(logits6)*h_std_o + h_mean_o
        # norm_6 = torch.norm(turnCplx(h_test_o[idx]) - test_tbh_cplx, dim=1)**2

        ### ISTA-Net
        # torch.cuda.empty_cache()
        logits7, _ = logits_net7((turnReal(turnCplx(y_test_h_o[idx])-h_mean_o)/h_std_o))
        test_tbh_cplx = turnCplx(logits7).to(device)*h_std_o + h_mean_o
        norm_7 = torch.norm(turnCplx(h_test_o[idx]) - test_tbh_cplx, dim=1)**2
        
        logits8, _ = logits_net8((turnReal(turnCplx(y_test_h_a[idx])-h_mean_a)/h_std_a))
        test_tbh_cplx = turnCplx(logits8).to(device)*h_std_a + h_mean_a
        norm_8 = torch.norm(turnCplx(h_test_a[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits9, _ = logits_net9((turnReal(turnCplx(y_test_h_o[idx])-h_mean_o)/h_std_o))
        # test_tbh_cplx = turnCplx(logits9).to(device)*h_std_o + h_mean_o
        # norm_9 = torch.norm(turnCplx(h_test_o[idx]) - test_tbh_cplx, dim=1)**2

        '''
        LS and LMMSE numerical solution for different IRS coefficient matrix
        '''

        # norm_LS_i, norm_LM_i = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test_o[idx], y_test_i[idx], IRS_coef_type='i')
        # norm_LS_d, norm_LM_d = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test_o[idx], y_test_d[idx], IRS_coef_type='d')
        norm_LS_h_o, norm_LM_h_o = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test_o[idx], y_test_h_o[idx], IRS_coef_type='h')
        norm_LS_h_a, norm_LM_h_a = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test_a[idx], y_test_h_a[idx], IRS_coef_type='h')


        NMSE_1[idx] = 10*torch.log10((norm_1 / torch.norm(h_test_o[idx], dim=1)**2).mean())
        NMSE_2[idx] = 10*torch.log10((norm_2 / torch.norm(h_test_a[idx], dim=1)**2).mean())
        # NMSE_3[idx] = 10*torch.log10((norm_3 / torch.norm(h_test_o[idx], dim=1)**2).mean())
        NMSE_4[idx] = 10*torch.log10((norm_4 / torch.norm(h_test_o[idx], dim=1)**2).mean())
        NMSE_5[idx] = 10*torch.log10((norm_5 / torch.norm(h_test_a[idx], dim=1)**2).mean())
        # NMSE_6[idx] = 10*torch.log10((norm_6 / torch.norm(h_test_o[idx], dim=1)**2).mean())
        NMSE_7[idx] = 10*torch.log10((norm_7 / torch.norm(h_test_o[idx], dim=1)**2).mean())
        NMSE_8[idx] = 10*torch.log10((norm_8 / torch.norm(h_test_a[idx], dim=1)**2).mean())
        # NMSE_9[idx] = 10*torch.log10((norm_9 / torch.norm(h_test_o[idx], dim=1)**2).mean())

    # NMSE_LS_i[idx] = 10*torch.log10((norm_LS_i / torch.norm(h_test_o[idx], dim=1)**2).mean())
    # NMSE_LM_i[idx] = 10*torch.log10((norm_LM_i / torch.norm(h_test_o[idx], dim=1)**2).mean())
    # NMSE_LS_d[idx] = 10*torch.log10((norm_LS_d / torch.norm(h_test_o[idx], dim=1)**2).mean())
    # NMSE_LM_d[idx] = 10*torch.log10((norm_LM_d / torch.norm(h_test_o[idx], dim=1)**2).mean())
    NMSE_LS_h_o[idx] = 10*torch.log10((norm_LS_h_o / torch.norm(h_test_o[idx], dim=1)**2).mean())
    NMSE_LM_h_o[idx] = 10*torch.log10((norm_LM_h_o / torch.norm(h_test_o[idx], dim=1)**2).mean())
    NMSE_LS_h_a[idx] = 10*torch.log10((norm_LS_h_a / torch.norm(h_test_a[idx], dim=1)**2).mean())
    NMSE_LM_h_a[idx] = 10*torch.log10((norm_LM_h_a / torch.norm(h_test_a[idx], dim=1)**2).mean())
    
    # plt.text(10*torch.log10(snr)-1,NMSE_LM_i[idx]-1, f'({NMSE_LM_i[idx].item():.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr)-1,NMSE_2[idx]-1, f'({10**(NMSE_2[idx].item()/10):.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr),NMSE_3[idx], f'({10**(NMSE_3[idx].item()/10):.2f})')  ## plot linear NMSE value 
    pbar.update(1)

    

# plt.plot(SNR_dB, NMSE_LS_i.to('cpu'), label='LS w/ I', linewidth=1, linestyle='--', marker='o', color="tab:blue")
# plt.plot(SNR_dB, NMSE_LS_d.to('cpu'), label='LS w/ D', linewidth=1, linestyle=':', marker='o', color="tab:blue")
plt.plot(SNR_dB, NMSE_LS_h_o.to('cpu'), label='LS w/ H ori', linewidth=1, linestyle=':', marker='o', color="tab:blue")
plt.plot(SNR_dB, NMSE_LS_h_a.to('cpu'), label='LS w/ H align', linewidth=1, linestyle='-', marker='o', color="tab:blue")
# plt.plot(SNR_dB, NMSE_LM_i.to('cpu'), label='LMMSE w/ I', linewidth=1, linestyle='--', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE_LM_d.to('cpu'), label='LMMSE w/ D', linewidth=1, linestyle=':', marker='o', color="tab:red")
plt.plot(SNR_dB, NMSE_LM_h_o.to('cpu'), label='LMMSE w/ H ori', linewidth=1, linestyle=':', marker='o', color="tab:red")
plt.plot(SNR_dB, NMSE_LM_h_a.to('cpu'), label='LMMSE w/ H align', linewidth=1, linestyle='-', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE5,'-x', label='model-based PD LS', linewidth=1 ,color="tab:red")
# plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')

plt.plot(SNR_dB, NMSE_4.to('cpu'), label='channelNet w/ H ori', linewidth=1, linestyle=':', marker='x', color="tab:green")  ###
plt.plot(SNR_dB, NMSE_5.to('cpu'), label='channelNet w/ H align', linewidth=1, linestyle='-', marker='x', color="tab:green")  ###
# plt.plot(SNR_dB, NMSE_6.to('cpu'), label='channelNet w/ H ', linewidth=1, linestyle='-', marker='x', color="tab:green")  ###

plt.plot(SNR_dB, NMSE_7.to('cpu'), label='ISTANet w/ H ori', linewidth=1, linestyle=':', marker='x', color="tab:brown")  ###
plt.plot(SNR_dB, NMSE_8.to('cpu'), label='ISTANet w/ H align', linewidth=1, linestyle='-', marker='x', color="tab:brown")  ###
# plt.plot(SNR_dB, NMSE_9.to('cpu'), label='ISTANet w/ H ', linewidth=1, linestyle='-', marker='x', color="tab:brown")  ###

plt.plot(SNR_dB, NMSE_1.to('cpu'), label='NP PD w/ H ori', linewidth=1, linestyle=':', marker='x', color="tab:orange")   ###
plt.plot(SNR_dB, NMSE_2.to('cpu'), label='NP PD w/ H align', linewidth=1, linestyle='-', marker='x', color="tab:orange")  ###
# plt.plot(SNR_dB, NMSE_3.to('cpu'), label='NP PD w/ H', linewidth=1, linestyle='-', marker='x', color="tab:orange")  ###


# plt.plot(SNR_dB, NMSE_3,'-x', label='channelNet w/o act. func.')  ###
# plt.plot(SNR_dB, NMSE_4,'-x', label='channelNet w/ act. func.')  ###

plt.suptitle("ChEsts NMSE with IRS vs SNR in UMa 28GHz")
# plt.title(' $[n_R,n_T]$:[4,36], MLP size:[288, 2048, 2048, 576] ')

# plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title('size:%s' %([2*n_R*T]+checkpoint1['hidden_sizes']+[2*n_R*n_T]))
plt.title('$[n_R,n_I,n_T,T]:[%s,%s,%s,%s], datasize: %s$' %(n_R,n_I,n_T,T,test_size))
plt.xlabel('SNR(dB)')
plt.ylabel('NMSE(dB)')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize='small')
plt.grid(True)
# plt.savefig('./simulation/result/snr/LS_mlp_Chest_-4_vs_i2.png')   ###
save_path = os.path.join(script_dir, 'snr', 'beam_align_%s_MLP_vs_CN_IN.pdf'%(channel)) #  %(IRS_coef_type)
plt.savefig(save_path)   ###

