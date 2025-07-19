import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

############ MAIN CONSOLE ############
mode = 'test'      # train or test
channel_model = 'InF'
version = '01'
sub_version = '01'
note = version + '_' + channel_model +'_' + sub_version
if channel_model == 'InF':
    load_training_file = './data/Training_Dataset_Eigen_InF25_Tis36.pt'
    load_testing_file = './data/Testing_Dataset_Eigen_InF25_Tis36.pt'
    scaling = 1e5
    n_layer = 5
    lamb = 8e-5
else:
    load_training_file = './data/Training_Dataset_Eigen_UMa28_Tis36.pt'
    load_testing_file = './data/Testing_Dataset_Eigen_UMa28_Tis36.pt'
    scaling = 1e5
    n_layer = 3
    lamb = 1.5e-5

model_name = 'model/model_parameters_ISTA_CNN_Net_' + note + '.pth'
fig_name = "fig/training_curve_CNN_" + note + ".png"
device = torch.device('cuda:2')
test_network = 1
test_conven = 0

######################################

def soft_threshold(v, c):
    q = torch.abs(v) - c
    return torch.sgn(v) * torch.max(q, torch.zeros_like(q))

class line_search():
    def __init__(self, A, lamb):
        self.A = A
        self.lamb = lamb

    def exact_line_search(self, grad):
        return torch.real((-grad.H @ -grad) / (-grad.H @ self.A.H @ self.A @ -grad))
    
    def Z(self, t, grad, x):
        return soft_threshold(x - t * grad, t*self.lamb)
    
    def f(self, y, x):
        return torch.norm(y - self.A @ x, 2)**2
    
    def f_hat(self, y, x, z, grad, t):
        return (self.f(y, x) + grad.H @ (z - x) + torch.norm(z - x, 2)**2 / (2*t)).real

    def back_tracking(self, y, grad, x, beta):
        t = 1
        z = self.Z(t, grad, x)
        f_z = self.f(y, z)
        f_hat = self.f_hat(y, x, z, grad, t)
        while f_z > f_hat:
            t = beta * t
            z = self.Z(t, grad, x)
            f_z = self.f(y, z)
            f_hat = self.f_hat(y, x, z, grad, t)
        return t

def ISTA_solver(y, A, lamb, beta=0.8, thres=1e-8, max_iter=2000):
    device = y.device
    MC = y.shape[1]
    len_x = A.shape[1]
    x_all = torch.zeros(len_x, MC, dtype=torch.complex64, device=device)
    LineSearch = line_search(A, lamb)
    
    total_count = 0
    total_nonzero = 0
    for m in range(MC):
        torch.manual_seed(2) 
        y_ = (y[:,m]).reshape(-1,1).to(torch.complex64)
        x_hat = torch.randn(len_x, 1, dtype=torch.complex64, device=device)
        total_res = []
        for k in range(max_iter):
            x_hat_old = x_hat
            grad = A.H @ ( A @ x_hat - y_ )
            # t = LineSearch.exact_line_search(grad)
            t = LineSearch.back_tracking(y_, grad, x_hat, beta)
            x_hat = soft_threshold(x_hat - t * grad, t*lamb)
            res = torch.norm(x_hat - x_hat_old)
            total_res.append(res.item())
            if res < thres: break
        if torch.norm(x_hat, 1) == 0: 
            print('Return zero vector')
            break
        x_all[:, m] = torch.squeeze(x_hat)
        total_count += k
        total_nonzero += len(x_hat[x_hat!=0])
    # print("\tlambda: {}\tnonzero: {}\tcount: {}".format(lamb, total_nonzero / MC, total_count/MC))
    avg_count = total_count / MC
    avg_nonzero = total_nonzero / MC
    return x_all, avg_count, avg_nonzero

def toppercent_matrix(A, percent):
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


def print_n_param(network, detail):
    if detail:
        for name, param in network.named_parameters():
            print(f"{name}: {param.size()} -> {param.numel()}")
    total = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Total learnable parameters: ', total)

def compute_chest_performance(hc_hat, h):
    hc_hat = to_matrix_form(hc_hat)
    h = to_matrix_form(h)
    M = h.shape[1]
    h_hat = U_tilda @ hc_hat
    loss = 0 
    for m in range(M):
        loss += (torch.norm(h_hat[:,m] - h[:,m])/torch.norm(h[:,m]))**2
    return (10*torch.log10(loss / M)).item()

def plot_loss(lossListTrain, lossListTest, fig_name):
    plt.clf()
    plt.plot(lossListTrain, label='training loss', color='blue')
    plt.plot(lossListTest, label='validation loss', color='red')
    plt.xlabel('epochs')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(fig_name)

def to_data_form(A):
    return torch.concat((A.real.T, A.imag.T), dim=1)

def to_matrix_form(A):
    return torch.complex(A[:,:int(A.shape[1]/2)], A[:,int(A.shape[1]/2):]).T

def LMMSE_solver(y, A, x_groundtruth, C_noise, MC):
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

def LS_solver(y, A):
    temp = torch.linalg.pinv(torch.conj(A.T) @ A) @ torch.conj(A.T)
    return temp @ y.to(torch.complex64)

def cov_mat(A, MC):
    mean_A = torch.reshape(torch.mean(A, dim=1), (-1, 1))
    mean_A_sqr = mean_A @ torch.conj(mean_A.T)
    C_A = 0
    for m in range(MC):
        m1 = torch.reshape(A[:,m], (-1, 1)) 
        m2 = torch.reshape(torch.conj(A[:,m]), (1, -1)) 
        C_A += (m1 @ m2 - mean_A_sqr)
    return C_A / MC

def gen_noisy_sgn(A, SNR):
    len, MC = A.shape
    A_pwr = torch.zeros(len, 1)
    for k in range(len):
        a = A[k,:].reshape(-1,1)
        A_pwr[k] = torch.real(a.H @ a / MC)
    noisepwr = A_pwr * 10 ** (-SNR/10)
    C_noise = torch.diag(torch.squeeze(noisepwr)).to(device)
    C_noise_stdev = torch.linalg.cholesky(C_noise).to(device)
    random = torch.complex(torch.randn(A.shape), torch.randn(A.shape)).to(device)
    noise =  (1/2**(1/2)) * C_noise_stdev.to(torch.complex128) @ random.to(torch.complex128)
    Y = A + noise
    return Y, C_noise

class conv1Dx(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super(conv1Dx, self).__init__()

        p = (k - 1) // 2 

        self.FR = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=1, padding=p)
        self.FI = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=1, padding=p)

    def forward(self, xR, xI):
        return self.FR(xR) - self.FI(xI), self.FI(xR) + self.FR(xI)
    
class densex(torch.nn.Module):
    def __init__(self, input_dim):
        super(densex, self).__init__()
        self.DR = nn.Linear(input_dim, input_dim)
        self.DI = nn.Linear(input_dim, input_dim)
    def forward(self, xR, xI):
        return self.DR(xR) - self.DI(xI), self.DI(xR) + self.DR(xI)

class BasicBlock01(torch.nn.Module):
    def __init__(self):
        super(BasicBlock01, self).__init__()

        self.t = nn.Parameter(torch.Tensor([0.5]))
        self.c = nn.Parameter(torch.Tensor([0.01]))

        self.conv1 = conv1Dx(1,8, k=5)
        self.conv2 = conv1Dx(8,16, k=5)
        self.conv3 = conv1Dx(16,8, k=5)
        self.conv4 = conv1Dx(8,1, k=5)
        self.conv5 = conv1Dx(1,8, k=5)
        self.conv6 = conv1Dx(8,16, k=5)
        self.conv7 = conv1Dx(16,8, k=5)
        self.conv8 = conv1Dx(8,1, k=5)
        self.relu = nn.ReLU()
        self.dense01 = densex(144)

    # method for each layer: pass through the layer and return x_k and loss_sym
    def forward(self, h_init, y):

        xR, xI = (h_init[:,:144]).unsqueeze(1), (h_init[:,144:]).unsqueeze(1)
        temp1R, temp1I = self.conv1(xR, xI)
        temp1R, temp1I = self.conv2(self.relu(temp1R), self.relu(temp1I))
        temp1R, temp1I = self.conv3(self.relu(temp1R), self.relu(temp1I))
        temp1R, temp1I = self.conv4(self.relu(temp1R), self.relu(temp1I))
        xR, xI = (y[:,:144]).unsqueeze(1), (y[:,144:]).unsqueeze(1)
        temp2R, temp2I = self.conv5(xR, xI)
        temp2R, temp2I = self.conv6(self.relu(temp2R), self.relu(temp2I))
        temp2R, temp2I = self.conv7(self.relu(temp2R), self.relu(temp2I))
        temp2R, temp2I = self.conv8(self.relu(temp2R), self.relu(temp2I))
        rR, rI = temp1R + temp2R, temp1I + temp2I
        rR, rI = self.dense01(rR, rI)
        
        r = torch.complex(rR.squeeze(1), rI.squeeze(1))
        h = torch.sgn(r) * F.relu(torch.abs(r) - self.c)
        h = torch.concat((h.real, h.imag), dim=1)

        return h
    
class ISTANet(torch.nn.Module):
    def __init__(self, n_layer):
        super(ISTANet, self).__init__()
        layers = []
        self.n_layer = n_layer  # define number of layers

        # concatenate layers according to the number of layers
        for _ in range(n_layer):
            layers.append(BasicBlock01())

        self.fcs = nn.ModuleList(layers)

    # method of ISTA-Net: initialize x_0, update x_k and collect loss_sym for each layer, and return x_final and aggregated loss_sym
    def forward(self, h_hat, y):

        # layers_sym = []   # for computing symmetric loss

        # for each layer: pass x_(k-1) to the layer k, and extract the x_pred and loss_sym for k-th layer
        for i in range(self.n_layer):
            h_hat = self.fcs[i](h_hat, y) # use method in BasicBlock and update x for each layer
            # print(compute_chest_performance(h_hat/scaling, h_all))
            # layers_sym.append(layer_sym)

        h_final = h_hat

        return h_final
    

torch.manual_seed(2) 
torch.set_printoptions(threshold=10000)

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Implement data loading logic here
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])

nTx = 36 # number of transmitters
nRx = 4 # number of receivers
SNR = [i for i in range(-4, 11, 2)]
# SNR = [i for i in range(6, 21, 3)]
# SNR = [0, 10]

note = str(nRx) + 'x' + str(nTx)

# sent signal (pilot)
factor = 1
X =  torch.eye(nTx, dtype=torch.cfloat)
X = X.repeat(1, factor)
X_tilda = torch.kron(X.T.contiguous(), torch.eye(nRx)).to(device)

if mode == 'train':
    Training_Dataset = torch.load(load_training_file)
    U_tilda = Training_Dataset['U_tilda']
    sparseChannel = Training_Dataset['SparseChan']      #########################################
    receivedSignal = Training_Dataset['ReceivedSig']
    groundChannel = Training_Dataset['GroundChan']
    n_samples = sparseChannel.shape[0]

    receivedSignal_Train, receivedSignal_Test, sparseChannel_Train, sparseChannel_Test = train_test_split(receivedSignal, sparseChannel, test_size=0.2, random_state=0)

    batch_size = 5000
    dataset_train = MyDataset((receivedSignal_Train, sparseChannel_Train))
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = MyDataset((receivedSignal_Test, sparseChannel_Test))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    len_hc = U_tilda.shape[1] # 144
    network = ISTANet(n_layer).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    epochs = 1000
    min_lr = 5e-7
    lossListTrain = []
    lossListTest = []
    mask = torch.eye(nRx*nTx).to(device)
    best_loss = float('inf')

    print("\n ===== Starting TRAINING =====")
    for e in range(epochs):
        avgTrainLoss = 0 
        network.train()
        for batch in tqdm(dataloader_train):

            receivedSignal, sparseChannel = batch
            receivedSignal = receivedSignal.to(torch.float32).to(device) * scaling
            sparseChannel = sparseChannel.to(torch.float32).to(device) * scaling

            batch_size = receivedSignal.shape[0]

            h_init = torch.randn((batch_size, len_hc*2), dtype=torch.float, device=device)    # compute x_0

            h_hat = network(h_init, receivedSignal)

            # Compute and print loss
            loss_discrepancy = torch.mean(torch.pow(h_hat - sparseChannel, 2))
            # loss_discrepancy = composite_loss(h_hat, sparseChannel, lam=0.95)
            loss_all = loss_discrepancy

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            avgTrainLoss += loss_all.item()
        
        avgTrainLoss = avgTrainLoss*(batch_size)/len(dataset_train)
        scheduler.step(avgTrainLoss)
        lossListTrain.append(avgTrainLoss)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            print(f"Stopping training as learning rate reached {current_lr:.2e}")
            break

        avgTestLoss = 0
        network.eval()
        with torch.no_grad():
            for batch in dataloader_test:

                receivedSignal, sparseChannel = batch  
                receivedSignal = receivedSignal.to(torch.float32).to(device) * scaling
                sparseChannel = sparseChannel.to(torch.float32).to(device) * scaling

                h_init = torch.randn((batch_size, len_hc*2), dtype=torch.float, device=device)

                h_hat = network(h_init, receivedSignal)

                # Compute and print loss
                loss_discrepancy = torch.mean(torch.pow(h_hat - sparseChannel, 2))
                # loss_discrepancy = composite_loss(h_hat, sparseChannel, lam=0.95)
                loss_all = loss_discrepancy
            
                avgTestLoss += loss_all.item()

        avgTestLoss = avgTestLoss*(batch_size)/len(dataset_test)
        if(best_loss>avgTestLoss):
            torch.save(network.state_dict(), model_name) 
            best_loss = avgTestLoss
        lossListTest.append(avgTestLoss)
        print("  Epoch {:d}\t| Training loss: {:.4e}\t | Testing loss: {:.4e} --- ratio: {:.3f}".format(e+1, avgTrainLoss, avgTestLoss, avgTrainLoss/avgTestLoss))
        if e % 2 == 0: 
            plot_loss(lossListTrain, lossListTest, fig_name)
    epochsList = np.linspace(0, epochs-1, num=epochs)
    plot_loss(lossListTrain, lossListTest, fig_name)

else:
    Testing_Dataset = torch.load(load_testing_file)
    groundChannel = Testing_Dataset['GroundChan'].to(device)
    receivedSignal = Testing_Dataset['ReceivedSig'].to(device)
    C_noise = Testing_Dataset['C_noise'].to(device)
    U_tilda = Testing_Dataset['U_tilda'].to(device)
    MC = int(groundChannel.shape[0]) # 3000

    U_tilda = U_tilda.to(device)
    h_all = to_matrix_form(groundChannel).to(device) # ([144, 3000])
    y_all = to_matrix_form(receivedSignal).to(device) # [144, 24000]

    len_hc = U_tilda.shape[1]
    network = ISTANet(n_layer).to(device)
    network.load_state_dict(torch.load(model_name))
    network = network.to(device)
    network.eval()
    print("\n=== Model successfully loaded ===")
    print_n_param(network, detail=0)

    LS_NMSE_all = torch.zeros(len(SNR), MC, device=device)
    LMMSE_NMSE_all = torch.zeros(len(SNR), MC, device=device)
    ISTA_NMSE_all = torch.zeros(len(SNR), MC, device=device)
    ISTA_CNN_Net_NMSE_all = torch.zeros(len(SNR), MC, device=device)

    print("\n===== Starting TESTING =====")

    counter1 = 0
    counter2 = 0
    for snr_ind in range(len(SNR)):

        print("==== SNR: {} dB ====".format(SNR[snr_ind]))

        y = y_all[:,counter1:counter1+3000] # [144, 3000]
        C_noise_k = C_noise[:,counter2:counter2+144]

        counter1 = counter1 + 3000
        counter2 = counter2 + 144

        h_lmmse = LMMSE_solver(y, X_tilda, h_all, C_noise_k, MC)
        h_ls = LS_solver(y, X_tilda)
        
        # LMMSE AND LS
        for m in range(MC):
            LMMSE_NMSE_all[snr_ind, m] = ((torch.norm(h_lmmse[:,m] - h_all[:,m], 2)/torch.norm(h_all[:,m], 2))**2).item()
            LS_NMSE_all[snr_ind, m] = ((torch.norm(h_ls[:,m] - h_all[:,m], 2)/torch.norm(h_all[:,m], 2))**2).item()

        if test_conven:
            # ISTA-LS
            if 1:
                XU = X_tilda @ U_tilda.to(torch.complex64) 
                hc_all_k, ISTA_count, nonzero = ISTA_solver(y, XU, lamb)
                h_all_k = U_tilda @ hc_all_k
                error_ista = h_all - h_all_k
                for m in range(MC):
                    ISTA_NMSE_all[snr_ind, m] = (torch.norm(error_ista[:,m])/torch.norm(h_all[:,m]))**2              

        if test_network: 
            
            # FIRST MODEL TESTING
            y_ = to_data_form(y).to(torch.float32) * scaling
            h_all_k = to_data_form(h_all)
            torch.manual_seed(74)
            hc_init = torch.randn((MC, len_hc*2), dtype=torch.float, device=device)
            hc_hat = network(hc_init, y_)
            hc_hat = to_matrix_form(hc_hat).to(U_tilda.dtype)
            count_nnz = torch.zeros(MC, 1)
            for m in range(MC):
                temp = toppercent_matrix(hc_hat[:,m], 0.95)
                count_nnz[m] = 144 - torch.count_nonzero(temp)
            h_hat = to_data_form(U_tilda @ hc_hat) / scaling
            error_ = to_matrix_form(h_all_k - h_hat)
            for m in range(MC):
                ISTA_CNN_Net_NMSE_all[snr_ind, m] = ((torch.norm(error_[:,m])/torch.norm(h_all[:,m]))**2).item()

        LMMSE_NMSE_k = torch.mean(LMMSE_NMSE_all[snr_ind, :])
        LS_NMSE_k = torch.mean(LS_NMSE_all[snr_ind, :])
        if test_conven: ISTA_NMSE_k = torch.mean(ISTA_NMSE_all, dim=1)[snr_ind]
        if test_network: ISTA_CNN_Net_NMSE_k = torch.mean(ISTA_CNN_Net_NMSE_all[snr_ind, :])

        print("\tLS ------------> NMSE:  {:.4f} dB ".format(10*torch.log10(LS_NMSE_k).item()))
        print("\tLMMSE ---------> NMSE:  {:.4f} dB ".format(10*torch.log10(LMMSE_NMSE_k).item()))
        if test_conven: print("\tISTA ----------> NMSE: {:.4f} dB \t non-zeros: {:.2f} \tCount: {:.2f}".format(10*torch.log10(ISTA_NMSE_k).item(), nonzero, ISTA_count))
        if test_network: 
            print("\tISTA-CNN-Net ---> NMSE: {:.4f} dB".format(10*torch.log10(ISTA_CNN_Net_NMSE_k).item()))
            print("\t\twith average sparsity of {:.2f}%".format(torch.mean(count_nnz).item()/144*100))
        
    LS_NMSE =torch.squeeze(torch.mean(LS_NMSE_all, dim=1)) 
    LMMSE_NMSE =torch.squeeze(torch.mean(LMMSE_NMSE_all, dim=1))
    if test_conven: ISTA_NMSE = torch.squeeze(torch.mean(ISTA_NMSE_all, dim=1))
    if test_network: ISTA_CNN_Net_NMSE = torch.squeeze(torch.mean(ISTA_CNN_Net_NMSE_all, dim=1))

    print("\n\n=== OVERALL SUMMARY ===")
    print("SNR = {};".format(list(SNR)))
    print("LS_NMSE = {};".format([round(e, 4) for e in (10*torch.log10(LS_NMSE)).tolist()]))
    print("LMMSE_NMSE = {};".format([round(e, 4) for e in (10*torch.log10(LMMSE_NMSE)).tolist()]))
    if test_conven: print("ISTA_NMSE = {};".format([round(e, 4) for e in (10*torch.log10(ISTA_NMSE)).tolist()]))
    if test_network: print("ISTA_CNN_Net_NMSE = {};".format([round(e, 4) for e in (10*torch.log10(ISTA_CNN_Net_NMSE)).tolist()]))

    print('\n====== END ======\n')