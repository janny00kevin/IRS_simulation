import torch
from torch.optim import SGD
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from utils.IRS_ct_channels_align import importData
from utils.complex_utils import turnReal, turnCplx, vec
from utils.NN_model.channelNet import channelNet
import os
from torch.optim.lr_scheduler import StepLR

def train(size, hidden_sizes, lr, num_epochs, batch_size, num_minibatch, cuda, snr, channel='UMa', IRS_coe_type='h', angle_config='x'):
    
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
        plt.suptitle("elbir MIMO ChEst with IRS coefficient %s" %IRS_coe_type)
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
    test_size = 2000
    SNR_dB = torch.tensor(list(range(min(snr),max(snr)+1,2))).to(device)
    SNR_lin = 10**(SNR_dB/10.0)
    print('training with SNR_dB:',SNR_dB)

    ## load training and testing data
    h, y, h_mean, h_std = importData(train_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type, phase='train', channel=channel, config=angle_config)
    h_test, y_test, _, _ = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type, phase='val', channel=channel, config=angle_config)
    Y_test_nmlz = (turnCplx(y_test).reshape(test_size, n_T*n_R, T//n_T) - h_mean)/h_std
    print('h_mean:',h_mean, 'h_std:',h_std)

    ## generate MLP ## initialize \tbh
    logits_net = channelNet(n_R, n_I, n_T, T).to(device)
    optimizer = SGD(logits_net.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # 3, 0.1 -> 10, 0.5

    ## initialize the training record
    num_iters = num_epochs*num_minibatch
    iter_loss = torch.zeros(num_iters).to(device)
    testing_loss = torch.zeros(num_iters).to(device)
    train_epochs = 0
    
    # Early stopping
    best_loss = float('inf')  # Initialize the best loss as infinity
    best_loss_epoch = float('inf')
    patience = 5
    epochs_without_improvement = 0

    ### training
    pbar = tqdm(total = num_iters)
    for i in range(num_epochs):
        torch.cuda.empty_cache()
        ### after 1 epoch, shuffle the dataset, using the same index to shuffle H and Y
        idx = torch.randperm(train_size).to(device)
        y = y[idx,:]
        h = h[idx,:]
        for j in range(num_minibatch):
            itr = j+i*num_minibatch+1
            ### trajectories training data
            tau_Y = turnCplx(y[j*batch_size:(j+1)*batch_size, :]).reshape(batch_size, n_T*n_R, T//n_T) ###
            tau_h = h[j*batch_size:(j+1)*batch_size, :]
            ### feed into the NN
            logits_net.train()
            logits = logits_net(torch.stack([tau_Y.real,tau_Y.imag,tau_Y.abs()],dim=1))
            
            ### compute loss and update
            optimizer.zero_grad()
            ## 2-norm square of y - \tbX\D\phi at each realization (size: num_trajectories by 1) 
            norm = torch.norm(tau_h - logits, dim=1)**2

            ## primal variable theta update
            loss_pri = norm.mean()
            loss_pri.backward()
            optimizer.step()

            with torch.no_grad():
                ## save the loss to plot
                iter_loss[itr-1] = (norm / torch.norm(tau_h, dim=1)**2).mean()
                
                ## validation
                logits_net.eval()
                logits_test = logits_net(torch.stack([Y_test_nmlz.real,Y_test_nmlz.imag,Y_test_nmlz.abs()],dim=1))
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

        train_epochs = i+1
        if best_loss < best_loss_epoch:
            best_loss_epoch = best_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print("epochs_without_improvement: ", epochs_without_improvement)
        if epochs_without_improvement >= patience:
            print("Early stopping triggered after %d epochs without improvement" % epochs_without_improvement)
            break
            
        scheduler.step()

    print('validation loss: ', iter_loss[itr-1].item())

    angle_config = 'o' if angle_config == 'original' else 'a' if angle_config == 'aligned_23.1' else angle_config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'trained_model', '%.3f_%s_elbir_psi_%s%s_lr%.0e_ep%s' 
            %(best_loss, channel, IRS_coe_type, angle_config, lr, train_epochs))
    lossPlot()
    parametersSave()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=1e-2) # learning rate
    parser.add_argument('-ep', type=int, default=1000)    # num of epochs
    parser.add_argument('-mbs', type=int, default=1000)  # size of mini-batch
    parser.add_argument('-nmb', type=int, default=1000)  # number of mini-batch 
    parser.add_argument('-nT', type=int, default=4)  
    parser.add_argument('-nI', type=int, default=8) 
    parser.add_argument('-nR', type=int, default=1)    
    parser.add_argument('-T', type=int, default=32)
    parser.add_argument('-hsz', type=int, default=[1], nargs='+')  # hidden layer size
    parser.add_argument('-cuda', type=int, default=0)  # cuda
    parser.add_argument('-snr', type=int, default=[-4,10], nargs='+')  
    parser.add_argument('-ch', type=str, default='default') 
    parser.add_argument('-psi', type=str, default='h') 
    parser.add_argument('-conf', type=str, default='x')   # ['original', 'o'], or ['aligned_23.1', 'a']
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available()) 

    torch.manual_seed(0)
    
    assert args.mbs*args.nmb == 1e6, 'mbs*nmb should be 1e6'
    
    train([args.nT, args.nR, args.nI, args.T], args.hsz, args.lr, args.ep, args.mbs, args.nmb, args.cuda, args.snr, args.ch, args.psi, args.conf)
