import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_path)))
from utils.batch_kronecker import batch_kronecker
from utils.IRS_ct_channels_8snr import import_data
from utils.complex_utils import turn_real, turn_cplx, vec
from utils.get_IRS_coef import get_IRS_coef
from utils.LMMSE import LMMSE_solver
import pandas as pd

torch.manual_seed(0)

device = torch.device("cuda:0")
# device1 = torch.device("cuda:2")
script_dir = os.path.dirname(os.path.abspath(__file__))
n_T = 4
n_I = 8
n_R = 4
T = 32

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-ch', type=str, default='default') 
args = parser.parse_args()
channel = args.ch
if channel == 'default':
    raise ValueError("Please specify a channel type using the -ch argument. Options are 'inf' or 'uma'.")
print('channel:', channel)

if channel == 'inf':
    # MLPs
    # filename1 = os.path.join(script_dir, 'trained_model', '4.646_eps1e+01_inf_AE_lr1e-03_[288, 1024, 288]_ep68.pt')
    # checkpoint1 = torch.load(filename1, weights_only=False, map_location=device)
    pass
    # filename2 = os.path.join(script_dir, 'trained_model', '1.619_eps10.0_inf_AE_psi_h_lr1e-03_[256, 1024, 258]_ep110.pt')
    # checkpoint2 = torch.load(filename2, weights_only=False, map_location=device)
    # filename3 = os.path.join(script_dir, 'trained_model', '1.282_8SNR_inf_5MLP_psi_h_lr5e-05_[256, 1024, 258]_ep129.pt')
    # checkpoint3 = torch.load(filename3, weights_only=False, map_location=device)


elif channel == 'uma':
    # MLPs
    # filename1 = os.path.join(script_dir, 'trained_model', '4.646_eps1e+01_inf_AE_lr1e-03_[288, 1024, 288]_ep68.pt')
    # checkpoint1 = torch.load(filename1, weights_only=False, map_location=device)
    pass
    # filename2 = os.path.join(script_dir, 'trained_model', '0.862_eps20.0_umaAE_psi_h_lr1e-03_[256, 1024, 258]_ep195.pt')
    # checkpoint2 = torch.load(filename2, weights_only=False, map_location=device)
    # filename3 = os.path.join(script_dir, 'trained_model', '0.720_8SNR_uma_5MLP_psi_h_lr1e-04_[256, 1024, 258]_ep170.pt')
    # checkpoint3 = torch.load(filename3, weights_only=False, map_location=device)



# logits_net1 = checkpoint1['logits_net'].to(device)
# logits_net2 = checkpoint2['logits_net'].to(device)
# logits_net3 = checkpoint3['logits_net'].to(device) 

sqrt2 = 2**.5
SNR_dB = torch.arange(-4,10.1,2)
SNR_lin = 10**(SNR_dB/10.0).to(device)
# NMSE_1 = torch.zeros_like(SNR_lin) # MLP 1
# NMSE_2 = torch.zeros_like(SNR_lin) # MLP 2
# NMSE_3 = torch.zeros_like(SNR_lin)

if channel == 'inf':
    # ISTA-LS-Net-conv
    # NMSE_1 = torch.tensor([7.2354, 4.5106, 3.3117, 3.0683, 2.8981, 2.6805, 2.7235, 2.6907])
    # nonparametric PD MLP
    NMSE_3 = torch.tensor([3.684815, 2.8516273, 1.1941444, 0.30232367, -0.3852597, -1.0081631, -0.74117315, -1.5960706])
    # channelNet
    NMSE_6 = torch.tensor([7.327077, 5.848506, 4.102896, 2.4392607, 0.7569467, -0.38965577, -0.58818984, -1.4034674])
    # ISTA-Net
    NMSE_9 = torch.tensor([6.3463726, 5.1957226, 4.1590524, 2.2825203, 0.9734659, 0.37809595, -0.012382846, -1.272726])
elif channel == 'uma':
    # NMSE_1 = torch.tensor([-16.4166, -18.578, -20.839, -22.5987, -24.5692, -26.3854, -28.0583, -30.4035])
    NMSE_3 = torch.tensor([1.5435733, -0.04704066, -0.43834013, -0.97817874, -1.8911169, -2.1430159, -3.4090219, -4.4230595])
    NMSE_6 = torch.tensor([4.4141884, 2.7210722, 1.3741971, -0.06888918, -1.3540541, -1.7496682, -2.7986841, -3.6328945])
    NMSE_9 = torch.tensor([3.8139496, 2.1144447, 1.2243052, 0.89034355, -0.123452395, -1.0715166, -1.7534541, -2.598897])
# NMSE_10 = torch.zeros_like(SNR_lin)

NMSE_LS_h = torch.zeros_like(SNR_lin) # LS
NMSE_LM_h = torch.zeros_like(SNR_lin) # sampled LMMSE

test_size = int(2.4e4)#3e3
datasize_per_SNR = test_size//len(SNR_lin)

### import testing data ###
h_test, y_test_i, h_mean, h_std = import_data(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='i', phase = 'test', channel=channel)
# print(h_mean, h_std)
_, y_test_d, _, _ = import_data(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='d', phase = 'test', channel=channel)
_, y_test_h, _, _ = import_data(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef='h', phase = 'test', channel=channel)
h_test = h_test.reshape(len(SNR_lin), datasize_per_SNR, n_R*n_T*n_I*2)
y_test_h = y_test_h.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)

def toppercent_matrix(A, percent):
    if torch.all(A == 0):
        return torch.zeros_like(A)
    # Step 1: Compute magnitude
    mag = torch.abs(A)

    # Step 2: Flatten and sort magnitudes (descending)
    mag_flat = mag.flatten()
    mag_sorted, _ = torch.sort(mag_flat, descending=True)

    # Step 3: Compute cumulative percentage
    total = torch.sum(mag_sorted)
    cum_sum = torch.cumsum(mag_sorted, dim=0)
    cum_percent = cum_sum / total

    # Step 4: Find the threshold that gives at least 90% cumulative sum
    idx_cut = torch.nonzero(cum_percent >= percent, as_tuple=True)[0][0]
    threshold = mag_sorted[idx_cut]

    # Step 5: Create mask and filter
    mask = mag >= threshold
    result = torch.zeros_like(A)
    result[mask] = A[mask]

    return result

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
    LS_solution = torch.matmul(tbPsi.pinverse(), turn_cplx(y_test).unsqueeze(2)).squeeze(2)
    # Compute the norm of the difference
    norm_LS = torch.norm((turn_cplx(h_test) - LS_solution), dim=1)**2

    ### LMMSE
    tbPsi = batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64)
    Sgnl = torch.matmul(tbPsi, turn_cplx(h_test).unsqueeze(2)).squeeze(2)
    Ps = (Sgnl.abs()**2).mean(dim=0)
    Pn = Ps / snr
    cov_n = torch.diag(Pn)
    lmmse = LMMSE_solver(turn_cplx(y_test).T, tbPsi, turn_cplx(h_test).T, cov_n, datasize_per_SNR).T
    norm_LM = torch.norm((turn_cplx(h_test) - lmmse), dim=1)**2
    return norm_LS,norm_LM

D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

pbar = tqdm(total = len(SNR_dB))
for idx, snr in enumerate(SNR_lin):
    snr = snr.unsqueeze(0).to(device)

    with torch.no_grad():
        # # Y_nmlz = (y.reshape(data_size,n_R,T)-h_mean2)/h_std2
        # logits1 = turn_cplx(logits_net1(turn_real(turn_cplx(y_test[idx])-h_mean)/h_std)[0])*h_std + h_mean
        # norm_1 = torch.norm(turn_cplx(h_test[idx]) - logits1, dim=1)**2

        logits2 = turn_cplx(logits_net2(turn_real(turn_cplx(y_test_h[idx])-h_mean)/h_std)[0])*h_std + h_mean
        norm_2 = torch.norm(turn_cplx(h_test[idx]) - logits2, dim=1)**2

        # if idx == len(SNR_lin) - 1:
        #     latent_sparse = logits_net2(turn_real(turn_cplx(y_test[idx])-h_mean)/h_std)[1]
        #     latent_cplx = torch.complex(latent_sparse[:, :144], latent_sparse[:, 144:])
        #     count_nnz = torch.zeros(datasize_per_SNR, 1)
        #     count_all0 = 0
        #     for m in range(datasize_per_SNR):
        #         temp = toppercent_matrix(latent_cplx.T[:,m], 0.95)
        #         if torch.all(temp == 0): 
        #             count_all0 += 1
        #         count_nnz[m] = 144 - torch.count_nonzero(temp)
        #     print("avg sparsity",torch.mean(count_nnz).item()/144*100, "%")
        #     print("all zero count", count_all0/datasize_per_SNR*100, "%")

        logits3 = logits_net3(turn_real(turn_cplx(y_test_h[idx])-h_mean)/h_std)
        test_tbh_cplx = turn_cplx(logits3)*h_std + h_mean
        norm_3 = torch.norm(turn_cplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        '''
        LS and LMMSE numerical solution for different IRS coefficient matrix
        '''
        norm_LS_h, norm_LM_h = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_h[idx], IRS_coef_type='h')



        # NMSE_1[idx] = 10*torch.log10((norm_1 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_2[idx] = 10*torch.log10((norm_2 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_3[idx] = 10*torch.log10((norm_3 / torch.norm(h_test[idx], dim=1)**2).mean())

    NMSE_LS_h[idx] = 10*torch.log10((norm_LS_h / torch.norm(h_test[idx], dim=1)**2).mean())
    NMSE_LM_h[idx] = 10*torch.log10((norm_LM_h / torch.norm(h_test[idx], dim=1)**2).mean())
    
    # plt.text(10*torch.log10(snr)-1,NMSE_LM_i[idx]-1, f'({NMSE_LM_i[idx].item():.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr)-1,NMSE_2[idx]-1, f'({10**(NMSE_2[idx].item()/10):.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr),NMSE_10[idx], f'({10**(NMSE_10[idx].item()/10):.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr),NMSE_10[idx], f'({(NMSE_10[idx].item()):.2f})')
    pbar.update(1)
print('channel:', channel)
print('LS:', [round(val, 4) for val in NMSE_LS_h.to('cpu').tolist()])
print('LMMSE:', [round(val, 4) for val in NMSE_LM_h.to('cpu').tolist()])
# print('ISTA-LS-Net-conv:', [round(val, 4) for val in NMSE_1.to('cpu').tolist()])
# print('parametric PD AE:', [round(val, 4) for val in NMSE_2.to('cpu').tolist()])
print('nonparametric PD MLP:', [round(val, 4) for val in NMSE_3.to('cpu').tolist()])
print('channelNet:', [round(val, 4) for val in NMSE_6.to('cpu').tolist()])
print('ISTANet:', [round(val, 4) for val in NMSE_9.to('cpu').tolist()])

plt.plot(SNR_dB, NMSE_LS_h.to('cpu'), label='LS', linewidth=1, linestyle='-', marker='o', color="tab:blue")
plt.plot(SNR_dB, NMSE_LM_h.to('cpu'), label='LMMSE', linewidth=1, linestyle='-', marker='o', color="tab:red")

plt.plot(SNR_dB, NMSE_6.to('cpu'), label='channelNet', linewidth=1, linestyle='-', marker='o', color="tab:green")  ###

plt.plot(SNR_dB, NMSE_9.to('cpu'), label='ISTANet', linewidth=1, linestyle='-', marker='o', color="tab:brown")  ###

# plt.plot(SNR_dB, NMSE_1.to('cpu'), label='ISTA-LS-Net-conv', linewidth=1, linestyle='-', marker='o', color="tab:pink")
plt.plot(SNR_dB, NMSE_3.to('cpu'), label='nonparametric PD MLP', linewidth=1, linestyle='-', marker='o', color="tab:orange")  ###
# plt.plot(SNR_dB, NMSE_1.to('cpu'), label='parametric PD AE $\\varepsilon=90$', linewidth=1, linestyle='-', marker='x', color="tab:gray")  ###
if channel == 'uma':
    plt.plot(SNR_dB, NMSE_2.to('cpu'), label='parametric PD AE $\\varepsilon=20$', linewidth=1, linestyle='-', marker='x', color="black")  ###
elif channel == 'inf':
    plt.plot(SNR_dB, NMSE_2.to('cpu'), label='parametric PD AE $\\varepsilon=10$', linewidth=1, linestyle='-', marker='x', color="black")


if channel == 'inf':
    plt.suptitle("NP PD trained MIMO nmlz ChEst vs SNR in InF2.5")
elif channel == 'uma':
    plt.suptitle("NP PD trained MIMO nmlz ChEst vs SNR in UMa28")
# plt.title(' $[n_R,n_T]$:[4,36], MLP size:[288, 2048, 2048, 576] ')

# plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title('size:%s' %([2*n_R*T]+checkpoint1['hidden_sizes']+[2*n_R*n_T]))
plt.title('$[n_R,n_I,n_T,T]:[%s,%s,%s,%s], datasize: %s$' %(n_R,n_I,n_T,T,test_size))
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE(dB)')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize='small')
plt.grid(True)

data = {
    "SNR_dB": SNR_dB.to('cpu'), "LS": NMSE_LS_h.to('cpu'), "LMMSE": NMSE_LM_h.to('cpu'),
    # "ISTA-LS-Net-conv": NMSE_1.to('cpu'), 
    "parametric PD AE": NMSE_2.to('cpu'), 
    "nonparametric PD MLP": NMSE_3.to('cpu'),
    "channelNet": NMSE_6.to('cpu'), "ISTANet": NMSE_9.to('cpu')
}
df = pd.DataFrame(data)
save_path = os.path.join(script_dir, 'ChEsts_testing_performance', '%s_MLP_vs_AE_.pdf'%(channel)) #  %(IRS_coef_type)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig(save_path, bbox_inches = 'tight')   ###
df.to_csv(os.path.splitext(save_path)[0] + '.csv', index=False)

