import torch
import os


device = torch.device("cuda:1")
script_dir = os.path.dirname(os.path.abspath(__file__))
n_T = 4
n_I = 8
n_R = 4
T = 32

filename1 = os.path.join(script_dir, 'trained_model', '1.594_eps5.0_inf_AE_psi_h_lr1e-03_[256, 1024, 256]_ep55.pt')
checkpoint1 = torch.load(filename1, weights_only=False, map_location=device)
logits_net1 = checkpoint1['logits_net'].to(device)

# channelNets
filename4 = os.path.join(script_dir, 'trained_model', '0.720_8SNR_uma_5MLP_psi_h_lr1e-04_[256, 1024, 258]_ep170.pt')
checkpoint4 = torch.load(filename4, weights_only=False, map_location=device)
logits_net4 = checkpoint4['logits_net'].to(device)


num_params = sum(p.numel() for p in logits_net1.parameters() if p.requires_grad)
print(f"Number of parameters in AE: {num_params}")

num_params = sum(p.numel() for p in logits_net4.parameters() if p.requires_grad)
print(f"Number of parameters in 5layers MLP: {num_params}")