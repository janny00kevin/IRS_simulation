import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch import randn

class BasicBlock(torch.nn.Module):
    def __init__(self, device, n_R, n_I, n_T, T):
        super(BasicBlock, self).__init__()

        self.device = device

        self.n_R = n_R
        self.n_I = n_I
        self.n_T = n_T
        self.T = T

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

    # method for each layer: pass through the layer and return x_k and loss_sym
    def forward(self, h_init, y):
        # compute r(k)
        # print('lambda:', self.lambda_step.device, 'y:', y.device)
        h = h_init.to(self.device) - self.lambda_step * h_init.to(self.device) + self.lambda_step * y.to(self.device)    # X_tilda is identity
        H_shape = self.n_R * self.n_I * self.n_T
        split_data = torch.stack((h[:,:H_shape], h[:,H_shape:]), dim=0)
        reshaped_data = split_data.view(2, -1, self.n_R*self.n_T, self.n_I)
        reshaped_data = reshaped_data.transpose(2,3)
        final_data = reshaped_data.permute(1, 0, 2, 3)
        h_input = final_data

        # FORWARD F(k): conv -> ReLU -> conv
        h = F.conv2d(h_input, self.conv1_forward, padding=1) # pass r(k) into conv layer
        h = F.relu(h) # pass ReLU
        h_forward = F.conv2d(h, self.conv2_forward, padding=1) # pass another conv layer

        # Soft-Thresholding
        # real_part = h_forward[:, 0, :, :] 
        # imag_part = h_forward[:, 1, :, :]
        # h_forward_ = torch.complex(real_part, imag_part)

        h = torch.mul(torch.sign(h_forward), F.relu(torch.abs(h_forward) - self.soft_thr))

        # BACKWARD F_inv(k): conv -> ReLU -> conv
        h = F.conv2d(h, self.conv1_backward, padding=1) # pass output from soft into conv layer
        h = F.relu(h) # pass ReLU
        h_backward = F.conv2d(h, self.conv2_backward, padding=1) # pass another conv layer

        # h_pred = h_backward.view(-1, 144) # x(k)
        h_back_real = h_backward[:,0,:,:]
        h_back_imag = h_backward[:,1,:,:]
        h_back_real = h_back_real.transpose(1,2).reshape(-1, H_shape)
        h_back_imag = h_back_imag.transpose(1,2).reshape(-1, H_shape)
        h_pred = torch.concat((h_back_real, h_back_imag),dim=1)

        # pass x_forward to BACK F_inv(k)
        h = F.conv2d(h_forward, self.conv1_backward, padding=1) # pass x_forward to the first conv layer in backward
        h = F.relu(h)
        h_est = F.conv2d(h, self.conv2_backward, padding=1)
        symloss = h_est - h_input # difference between r(k) [before FORWARD] and F_inv(F(r(k)))

        return [h_pred, symloss]

class ISTANet(torch.nn.Module):
    def __init__(self, LayerNo, device, n_R, n_I, n_T, T):
        super(ISTANet, self).__init__()
        self.device = device
        onelayer = []
        self.LayerNo = LayerNo  # define number of layers

        # concatenate layers according to the number of layers
        for i in range(LayerNo):
            onelayer.append(BasicBlock(device, n_R, n_I, n_T, T))

        self.fcs = nn.ModuleList(onelayer)

    # method of ISTA-Net: initialize x_0, update x_k and collect loss_sym for each layer, and return x_final and aggregated loss_sym
    def forward(self, y):

        h_hat = randn((y.shape), dtype=torch.float).to(self.device)    # compute x_0

        layers_sym = []   # for computing symmetric loss

        # for each layer: pass x_(k-1) to the layer k, and extract the x_pred and loss_sym for k-th layer
        for i in range(self.LayerNo):
            [h_hat, layer_sym] = self.fcs[i](h_hat, y) # use method in BasicBlock and update x for each layer
            layers_sym.append(layer_sym)

        h_final = h_hat

        return [h_final, layers_sym]