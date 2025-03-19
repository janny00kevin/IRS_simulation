import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from utils.IRS_ct_channels import importData
from utils.complex_utils import turnReal, turnCplx, vec
from utils.NN_model.ISTANet import ISTANet
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(size, hidden_sizes, lr, num_epochs, batch_size, num_minibatch, cuda, snr, channel='UMa', IRS_coe_type='h', num_layers=10):
    
    def parametersSave():
    ## save the model and parameters
        checkpoint = {
            'logits_net': best_model,
            'num_epochs': num_epochs, 'train_epochs': train_epochs, 'num_minibatch': num_minibatch, 'test_size': test_size,
            'n_R': n_R, 'n_T': n_T, 'n_I': n_I, 'T': T, 'lr' : lr, 'SNR_dB': SNR_dB,
            'iter_loss': iter_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"),
            'testing_loss': testing_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), 
            'h_mean': h_mean, 'h_std': h_std,
            'IRS_coe_type': IRS_coe_type
        }
        
        torch.save(checkpoint, save_path + '.pt')
    
    def NMSEplot(epochs):
        plt.subplot(311)
        plt.plot(epochs, iter_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training NMSE')
        plt.plot(epochs, testing_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation NMSE')
        plt.suptitle("nonparametric PD trained MMSE MIMO ChEst with IRS coefficient %s" %IRS_coe_type)
        plt.title(' $[n_R,n_I,n_T,T]$:[%s,%s,%s,%s], lr:%s, size:%s, SNR:%s~%s' 
                  %(n_R,n_I,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T*n_I+2], torch.min(SNR_dB).item(), torch.max(SNR_dB).item()))
        
        plt.xlabel('epochs')
        plt.legend()
        plt.grid(True)
    
    def lossPlot():
        ## plot the loss
        epochs = range(1, train_epochs+1)
        NMSEplot(epochs)

        plt.savefig(save_path + '.pdf')
        plt.close()

    def grad_norm(logits_net):
        gradient_norm = 0
        for _, param in logits_net.named_parameters():
            gradient_norm += torch.norm(param.grad)**2
        return gradient_norm
    
    device = torch.device("cuda:%s"%(cuda) if torch.cuda.is_available() else "cpu")
    n_T, n_R, n_I, T = size
    train_size = int(num_minibatch*batch_size)
    print('num_minibatch', type(num_minibatch), 'batch_size', type(batch_size), 'train_size', type(train_size))
    test_size = 2000
    SNR_dB = torch.tensor(list(range(min(snr),max(snr)+1,2))).to(device)
    SNR_lin = 10**(SNR_dB/10.0)
    print('training with SNR_dB:',SNR_dB)

    ## load training and testing data
    # print('train_size', type(train_size), 'n_R', type(n_R), 'n_I', type(n_I), 'n_T', type(n_T), 'T', type(T))
    h, y, h_mean, h_std = importData(train_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type, phase = 'train', channel=channel)
    h_test, y_test, _, _ = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type, phase = 'val', channel=channel)
    print('h_mean:',h_mean, 'h_std:',h_std)

    ## build the model
    logits_net = ISTANet(num_layers, device, n_R, n_I, n_T, T).to(device)
    optimizer = Adam(logits_net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.1, threshold=1e-3)
    current_lr = optimizer.param_groups[0]['lr']

    ## initialize the training record
    num_iters = num_epochs*num_minibatch
    iter_loss = torch.zeros(num_iters).to(device)
    testing_loss = torch.zeros(num_iters).to(device)
    train_epochs = 0

    # Early stopping
    best_loss = float('inf')  # Initialize the best loss as infinity
    min_lr = 5e-7  # Minimum learning rate
    best_model = None  # Initialize the best model as None

    ### training
    pbar = tqdm(total = num_iters)
    for i in range(num_epochs):
        train_epochs = i+1
        torch.cuda.empty_cache()
        idx = torch.randperm(train_size)
        y.copy_(y[idx, :])
        h.copy_(h[idx, :])
        for j in range(num_minibatch):
            itr = j+i*num_minibatch+1
            ### extract the mini-batch
            tau_y = y[j*batch_size:(j+1)*batch_size, :].to(device)
            tau_h = h[j*batch_size:(j+1)*batch_size, :].to(device)

            ### feed into the NN
            logits_net.train()
            logits, layers_sym = logits_net(tau_y)
            
            ### compute loss and update
            optimizer.zero_grad()
            ## 2-norm square of y - \tbX\D\phi at each realization (size: batch_size by 1) 
            norm = torch.norm(tau_h - logits, dim=1)**2

            ## primal variable theta update
            loss_constraint = (torch.stack(layers_sym).reshape(num_layers,batch_size,2*n_T*n_I*n_R).norm(dim=-1)**2).sum(dim=0)
            loss_func = torch.mean(norm + 0.01 * loss_constraint)
            loss_func.backward()
            optimizer.step()

            with torch.no_grad():
                ## save the loss to plot
                iter_loss[itr-1] = (norm / torch.norm(tau_h, dim=1)**2).mean()
                
                ## validation
                logits_net.eval()
                logits_test, _ = logits_net(turnReal(turnCplx(y_test)-h_mean)/h_std)
                test_tbh_cplx = turnCplx(logits_test)*h_std + h_mean
                norm_test = torch.norm(turnCplx(h_test) - test_tbh_cplx, dim=1)**2
                testing_loss[itr-1] = (norm_test / torch.norm(h_test, dim=1)**2).mean()
                loss_n4 = (norm_test / torch.norm(h_test, dim=1)**2)[:test_size//len(SNR_dB)//2].mean()
                loss_10 = (norm_test / torch.norm(h_test, dim=1)**2)[test_size-test_size//len(SNR_dB)//2:].mean()

                if (testing_loss[itr-1] < best_loss):
                    best_loss = testing_loss[itr-1].item()
                    best_model = logits_net

            pbar.set_description('tNMSE:%s, vNMSE:%s, -4:%s, 10:%s, g:%s' 
                    %(format(float(iter_loss[itr-1]), '.3f'), format(float(testing_loss[itr-1]), '.3f'),
                      format(10*loss_n4.log10(), '.3f'), format(10*loss_10.log10(), '.3f'), format((grad_norm(logits_net)), '.2e')))
            pbar.update(1)

        scheduler.step(testing_loss[i*num_minibatch:(i+1)*num_minibatch].mean())
        if current_lr != optimizer.param_groups[0]['lr']:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {i+1}: Learning rate is {current_lr:.1e}")
        if current_lr < min_lr:
            print(f"Stopping training as learning rate reached {current_lr:.2e} at epoch {i+1}")
            break

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'trained_model', '%.3f_%s_ISTA_psi_%s_lr%.0e_ep%s' 
            %(best_loss, channel, IRS_coe_type, lr, train_epochs))
    lossPlot()
    parametersSave()
    print('validation loss: ', iter_loss[itr-1].item())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=1e-3) # learning rate
    parser.add_argument('-ep', type=int, default=1000)    # num of epochs
    parser.add_argument('-mbs', type=int, default=1000)  # size of mini-batch
    parser.add_argument('-nmb', type=int, default=1000)  # number of mini-batch
    parser.add_argument('-nT', type=int, default=4)  
    parser.add_argument('-nI', type=int, default=8)   
    parser.add_argument('-nR', type=int, default=4)   
    parser.add_argument('-T', type=int, default=32)
    parser.add_argument('-hsz', type=int, default=[1], nargs='+')  # hidden layer size
    parser.add_argument('-cuda', type=int, default=0)  # cuda
    parser.add_argument('-snr', type=int, default=[-4,10], nargs='+')  
    parser.add_argument('-ch', type=str, default='default') 
    parser.add_argument('-psi', type=str, default='identity') 
    parser.add_argument('-lys', type=int, default=10)  # num of layers of ISTA-Net
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available()) 

    torch.manual_seed(0)

    if args.nmb == 1:
        nmb = int(1e6 // args.mbs)
        print('mbs:', type(args.mbs))
    else:
        nmb = args.nmb
    # print("num of mini-batch: ", nmb, "mini-batch size: ", args.mbs)
    
    train([args.nT, args.nR, args.nI, args.T], args.hsz, args.lr, args.ep, args.mbs, nmb, args.cuda, args.snr, args.ch, args.psi, args.lys)
