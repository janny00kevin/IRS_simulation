import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch import randn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
# from network_ISTA_NET_ChEst import *
from utils.NN_model.ISTANet import ISTANet
# from sklearn.model_selection import train_test_split
# import mat73
# import h5py
from utils.IRS_rician_channel import importData
from utils.complex_utils import turnReal, turnCplx, vec

torch.manual_seed(0) 
torch.set_printoptions(threshold=10000)
device = torch.device('cuda:2')
IRS_coe_type = 'i'

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Implement data loading logic here
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])

def cMM(A,B):
    # A, B = AA.to(torch.complex128), BB.to(torch.complex128)
    Ar, Ai = A[:,:int(A.shape[1]/2)].to(torch.float32), A[:,int(A.shape[1]/2):].to(torch.float32)
    Br, Bi = B[:,:int(B.shape[1]/2)].to(torch.float32), B[:,int(B.shape[1]/2):].to(torch.float32)
    out_r = Ar @ Br - Ai @ Bi
    out_i = Ar @ Bi + Ai @ Br
    return torch.concat((out_r, out_i),dim=1)

def cmplx2concat(A):
    return torch.concat((A.real, A.imag), dim=1)

def mergetocmplx(A):
    return torch.complex(A[:,:int(A.shape[1]/2)], A[:,int(A.shape[1]/2):])

# nTx = 36 # number of transmitters
# nRx = 4 # number of receivers
SNR_dB = torch.tensor(list(range(-4,11,2))).to(device)
SNR_lin = 10**(SNR_dB/10.0)
n_R = 4
n_T = 4
T = 32
n_I = 8

# note = str(nRx) + 'x' + str(nTx)

# sent signal (pilot)
# factor = 1
# X =  torch.eye(nTx, dtype=torch.cfloat)
# X = X.repeat(1, factor)
# X_tilda = torch.kron(X.T.contiguous(), torch.eye(nRx))
# X_H = X_tilda.H
# X_H_X = cMM(cmplx2concat(X_tilda.H), cmplx2concat(X_tilda))
# X_H_X = mergetocmplx(X_H_X)

# load_training_file = './data/A_Training_Dataset_ADR_RayTracing_G15_KTR_TisnT.mat'
# try:
#     Training_Dataset = mat73.loadmat(load_training_file)
#     U_tilda = torch.tensor(Training_Dataset['A_tilda'])
#     sparseChannel = Training_Dataset['SparseChan']
#     receivedSignal = Training_Dataset['ReceivedSig']
#     n_samples = sparseChannel.shape[0]
# except:
#     with h5py.File(load_training_file, "r") as f:
#         receivedSignal = f["ReceivedSig"][:]
#         sparseChannel = f["SparseChan"][:]
#         groundChannel = f["GroundChan"][:]
#         U_tilda = f["A_tilda"][:]

#     receivedSignal = torch.tensor(receivedSignal)
#     sparseChannel = torch.tensor(sparseChannel)
#     groundChannel = torch.tensor(groundChannel)
#     U_tilda = torch.tensor(U_tilda)
#     n_samples = sparseChannel.shape[0]

train_size = int(1e6)
test_size = int(2e3)
## load training and testing data
groundChannel_Train, receivedSignal_Train, _, _ = importData(train_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type, case = 'train')
groundChannel_Test, receivedSignal_Test, _, _ = importData(test_size, n_R, n_I, n_T, T, 10**(torch.tensor([10]).to(device)/10.0), device, IRScoef=IRS_coe_type, case = 'train')

# training dataset
# Training_Dataset = mat73.loadmat('./data/A_Testing_Dataset_ADR_RayTracing_5L_TisnT_3000.mat')
# U_tilda = torch.tensor(Training_Dataset['A_tilda'])
# sparseChannel = Training_Dataset['SparseChan']
# receivedSignal = Training_Dataset['ReceivedSig']
# groundChannel = Training_Dataset['GroundChan']
# n_samples = sparseChannel.shape[0]

# a = mergetocmplx(torch.tensor(sparseChannel)).T.to(U_tilda.dtype)
# b = U_tilda @ a
# groundChannel = cmplx2concat(b.T)

# normalize the datset 
# sparseChannel_ = mergetocmplx(torch.tensor(sparseChannel)).T
# receivedSignal_ = mergetocmplx(torch.tensor(receivedSignal)).T
# sparse_stdnrm, sparse_mean, sparse_std = std_normalization(sparseChannel_)
# received_stdnrm, received_mean, received_std = std_normalization(receivedSignal_)
# # sparseChannel = cmplx2concat(sparse_stdnrm.T)
# received_stdnrm_ = cmplx2concat(received_stdnrm.T)   

# split the data into training and test sets
# receivedSignal_Train, receivedSignal_Test, groundChannel_Train, groundChannel_Test = train_test_split(torch.tensor(receivedSignal), groundChannel, test_size=0.2, random_state=0)
# receivedSignal_Train, sparseChannel_Train = receivedSignal, sparseChannel

batch_size_train = int(1e3)
dataset_train = MyDataset((receivedSignal_Train, groundChannel_Train))
dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)

batch_size_test = int(2e3)
dataset_test = MyDataset((receivedSignal_Test, groundChannel_Test))
dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)

# scaling = 1e5
n_layer = 10
# loss_function = 'MSE'      # 'MSE', 'Avg MSE'
# alpha = 10
network = ISTANet(n_layer, device, n_R, n_I, n_T, T).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.1)
epochs = 3000
min_lr = 5e-7
# lossListTrain = []
# lossListTest = []
# mask = torch.eye(nRx*nTx).to(device)
best_loss = float('inf')
current_lr = optimizer.param_groups[0]['lr']

# model_name = 'model/model_parameters_ADR_RayTracing_ISTA_Net_G15_01.pth'

# network.load_state_dict(torch.load(model_name))
# network = network.to(device)

print("Starting training: ")
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# pbar = tqdm(total = epochs*(train_size//batch_size_train))
for e in range(epochs):
    avgTrainLoss = 0 
    network.train()
    for batch in tqdm(dataloader_train):

        receivedSignal, groundChannel = batch
        receivedSignal = receivedSignal.to(torch.float32).to(device)
        groundChannel = groundChannel.to(torch.float32).to(device)

        h_hat, loss_layers_sym = network(receivedSignal)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(h_hat - groundChannel, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(n_layer-1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        gamma = torch.Tensor([0.01]).to(device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        # optimizer.zero_grad()

        # # consider only conventional MSE
        # loss_ = loss[-1] / batch_size
        # loss_.backward(retain_graph=True)
        
        # # # Clip gradients
        # max_norm = 1.0  # Define the maximum gradient norm
        # nn.utils.clip_grad_norm_(network.parameters(), max_norm)

        # optimizer.step() 

        avgTrainLoss += loss_all.item()
    
    avgTrainLoss = avgTrainLoss*(batch_size_train/len(dataset_train))
    scheduler.step(avgTrainLoss)
    # lossListTrain.append(avgTrainLoss)
    # print('Epoch %d | Loss %2.12f' % (e, avgTrainLoss))

    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != optimizer.param_groups[0]['lr']:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {e+1}: Learning rate is {current_lr:.1e}")
    if current_lr < min_lr:
        print(f"Stopping training as learning rate reached {current_lr:.2e}")
        break

    avgTestLoss = 0
    network.eval()
    with torch.no_grad():
        for batch in dataloader_test:

            receivedSignal, groundChannel = batch  
            receivedSignal = receivedSignal.to(torch.float32).to(device)
            groundChannel = groundChannel.to(torch.float32).to(device)

            h_hat, loss_layers_sym = network(receivedSignal)

            # Compute and print loss
            loss_discrepancy = torch.mean(torch.pow(h_hat - groundChannel, 2))

            loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
            for k in range(n_layer-1):
                loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

            gamma = torch.Tensor([0.01]).to(device)

            # loss_all = loss_discrepancy
            loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
        
            avgTestLoss += loss_all.item()

        receivedSignal, groundChannel = next(iter(dataloader_test))
        h_hat, _ = network(receivedSignal.to(torch.float32))
        testing_loss = (torch.norm(groundChannel - h_hat, dim=1)**2 / torch.norm(groundChannel, dim=1)**2).mean()
        # testing_loss = torch.mean(torch.pow(receivedSignal - groundChannel, 2))/ torch.norm(groundChannel, dim=1)**2
    avgTestLoss = avgTestLoss*(batch_size_test/len(dataset_test))
    if(best_loss>avgTestLoss):
        best_model = network
        best_loss = avgTestLoss
    # lossListTest.append(avgTestLoss)
    # print(avgTestLoss)
    print("  Epoch {:d}\t| Training loss: {:.4e}\t | Testing loss: {:.4e} | Testing NMSE (snr: -10): {:.3f}".format(e+1, avgTrainLoss, avgTestLoss, 10*testing_loss.log10()))
    train_epochs = e+1
# epochsList = np.linspace(0, epochs-1, num=epochs)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'result', '%.3f_ric_nattISTA_nmlz_psi_%s_ep%s' %(best_loss, IRS_coe_type, train_epochs))        
checkpoint = {'logits_net': best_model,}
torch.save(checkpoint, save_path + '.pt')

# plt.clf()
# fig, ax = plt.subplots()
# ax.plot(epochsList, lossListTrain, color='blue', label='Training Loss')
# ax.plot(epochsList, lossListTest, color='red', label='Testing Loss')
# ax.set_xlabel('epoch')
# ax.set_ylabel('Loss')
# ax.set_title('Training and Testing Curves')
# ax.legend()
# plt.savefig("Training_and_Testing_Curves.jpg")

# plot_until = 100
# plt.clf()
# fig, ax = plt.subplots()
# ax.plot(epochsList[:plot_until], lossListTrain[:plot_until], color='blue', label='Training Loss')
# ax.plot(epochsList[:plot_until], lossListTest[:plot_until], color='red', label='Testing Loss')
# ax.set_xlabel('SNR [dB]')
# ax.set_ylabel('Loss')
# ax.set_title('Training and Testing Curves in First 100 epoch')
# ax.legend()
# plt.savefig("Training_and_Testing_Curves_100.jpg")