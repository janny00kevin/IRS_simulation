import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
from tqdm import tqdm
# import h5py
from utils.batch_khatri_rao import batch_khatri_rao
from utils.IRS_rayleigh_channel import importData
from utils.complex_utils import turnReal, turnCplx, vec
import os


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def train(size, hidden_sizes=[64, 32], lr=1e-3, num_epochs = 3, 
        num_trajectories = 1e4, num_minibatch = 10, cuda = 1, snr = 0, channel = 'UMa', IRS_coe_type = 'i'):
    
    def parametersSave():
    ## save the model and parameters
        checkpoint = {
            'logits_net': logits_net,
            'hidden_sizes': hidden_sizes,
            # 'num_channel': num_channel,
            # 'filt_size': filt_size,
            'num_epochs': num_epochs,
            'train_epochs': train_epochs,
            'num_minibatch': num_minibatch,
            'test_size': test_size,
            'n_R': n_R,
            'n_T': n_T,
            'n_I': n_I,
            'T': T,
            'lr' : lr,
            'lr_l': lr_l,
            # 'lr_t': lr_t,
            # 'H_mean': H_mean,
            # 'H_sigma' :H_sigma,
            # 'W_Mean': W_Mean,
            'SNR_dB': SNR_dB,
            't_rec': t_rec,
            'lamb_rec': lamb_rec,
            'iter_loss': iter_loss.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"),
            'testing_loss': testing_loss.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), 
            'h_mean': h_mean,
            'h_std': h_std,
            'IRS_coe_type': IRS_coe_type
        }
        # torch.save(checkpoint, './simulation/result/SP_UMa_lr%s_%s_ep%s_SNR%s.pt' %(lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T+2],train_epochs,SNR_dB.tolist()))
        
        save_path = os.path.join(script_dir, 'result', '%.3f_SP_ray_psi_%s_lr%.0e_%s_ep%s.pt' 
            %(testing_loss[itr-1].item(), IRS_coe_type, lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T*n_I+2], train_epochs))
        torch.save(checkpoint, save_path)
    
    def NMSEplot(epochs):
        plt.subplot(311)
        plt.plot(epochs, iter_loss.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training nmse')
        plt.plot(epochs, testing_loss.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation nmse')
        plt.suptitle("nonparametric PD trained MMSE MIMO ChEst with IRS coefficient %s" %IRS_coe_type)
        plt.title(' $[n_R,n_I,n_T,T]$:[%s,%s,%s,%s], lr:%s, size:%s, SNR:%s~%s' 
                  %(n_R,n_I,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T*n_I+2], torch.min(SNR_dB).item(), torch.max(SNR_dB).item()))
        
        plt.xlabel('epochs')
        # plt.ylabel('mse')
        plt.legend()
        plt.grid(True)

    def t_MSE_Plot(epochs):
        plt.subplot(312)
        plt.plot(epochs,t_rec.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='t')
        # plt.plot(epochs,lse.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"),label=r'$\|\mathbf{y} - \tilde{\mathbf{X}}\mathbf{h}\|_2^2$')
        plt.xlabel('epochs')
        plt.grid(True)
        plt.legend()

    def lambPlot(epochs):
        plt.subplot(313)
        plt.plot(epochs,lamb_rec.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"),label='$\\lambda$')
        plt.xlabel('epochs')
        plt.grid(True)
        plt.legend()
    
    def lossPlot():
        ## plot the loss
        epochs = range(1, train_epochs+1)
        NMSEplot(epochs)
        t_MSE_Plot(epochs)
        lambPlot(epochs)

        save_path = os.path.join(script_dir, 'result', '%.3f_SP_ray_psi_%s_lr%.0e_%s_ep%s.pdf' 
                    %(testing_loss[itr-1].item(), IRS_coe_type, lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T*n_I+2], train_epochs))
        plt.savefig(save_path)
        plt.close()

    def grad_norm(logits_net):
        gradient_norm = 0
        for _, param in logits_net.named_parameters():
            gradient_norm += torch.norm(param.grad)**2
        return gradient_norm
    
    def termination(logits_net, priFeasibility, lambda_val, epsilon):
        # calculate_norm(y, X_tilde, D, h_tilde) < epsilon or k == num_iterations-1
        gradient_norm = grad_norm(logits_net)
        return gradient_norm <= epsilon and priFeasibility <= 0 and lambda_val.real >= 0 and lambda_val.real*priFeasibility <= epsilon
    
    device = torch.device("cuda:%s"%(cuda) if torch.cuda.is_available() else "cpu")
    n_T, n_R, n_I, T = size
    train_size = num_minibatch*num_trajectories
    # channel_information = [0, 1, 0, 0.1]
    # H_mean, H_sigma, W_Mean = channel_information
    test_size = 2000                                                       
    SNR_dB = torch.tensor(list(range(min(snr),max(snr)+1,2))).to(device)   ###
    SNR_lin = 10**(SNR_dB/10.0)
    print('SNR_dB:',SNR_dB)
    # print('channel:',channel)

    ## load training and testing data
    h, y, h_mean, h_std = importData(train_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type)
    h_test, y_test, _, _ = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type)
    print('h_mean:',h_mean, 'h_std:',h_std)
    # Y_test = (torch.view_as_complex(y_test.reshape(test_size,n_R,T,2))-h_mean)/h_std
    c = torch.cat((torch.zeros(n_R*n_T*n_I,1), torch.ones(1,1))).to(torch.complex64).to(device)
    D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

    ## initialize lambda   
    lamb = torch.tensor([1.0], requires_grad=True, device=device)
    lr_l = 1e-4 #5e-5                                                    ###
    epsilon = 1e-16

    ## generate MLP ## initialize \tbh
    logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*n_R*n_T*n_I+2]).to(device)
    optimizer_pri = Adam(logits_net.parameters(), lr=lr)

    ## initialize the training record
    num_iters = num_epochs*num_minibatch
    iter_loss = torch.zeros(num_iters).to(device)
    testing_loss = torch.zeros(num_iters).to(device)
    t_rec = torch.zeros(num_iters).to(device)
    lamb_rec = torch.zeros(num_iters).to(device)
    train_epochs = 0

    ### trainning
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
            tau_y = y[j*num_trajectories:(j+1)*num_trajectories, :]
            tau_h = h[j*num_trajectories:(j+1)*num_trajectories, :]
            ### feed into the NN
            logits = logits_net(tau_y)
            tau_tbh_cplx = turnCplx(logits)
            # tau_h_hat = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
            
            ### compute loss and update
            optimizer_pri.zero_grad()
            ## 2-norm square of y - \tbX\D\phi at each realization (size: num_trajectories by 1) 
            norm = torch.norm(turnCplx(tau_h) - D.matmul(tau_tbh_cplx.T).T, dim=1)**2

            lamb_dt = lamb.detach().clone()

            ## primal variable theta update
            loss_pri = ((1-lamb_dt)*c.T.matmul(tau_tbh_cplx.T).real + lamb_dt*norm).mean()
            loss_pri.backward()
            optimizer_pri.step()
            
            ## dual variable lambda update
            lamb = lamb + lr_l/itr * torch.mean(norm-c.T.matmul(tau_tbh_cplx.T).real)
            if lamb.item() < 0:
                lamb.data.fill_(0)

            with torch.no_grad():
                ## save the loss to plot
                iter_loss[itr-1] = (norm / torch.norm(tau_h, dim=1)**2).mean()
                
                ## validation
                logits_test = logits_net(y_test)
                test_tbh_cplx = turnCplx(logits_test)
                norm_test = torch.norm(turnCplx(h_test) - D.matmul(test_tbh_cplx.T).T, dim=1)**2
                testing_loss[itr-1] = (norm_test / torch.norm(h_test, dim=1)**2).mean()
                loss_n4 = (norm_test / torch.norm(h_test, dim=1)**2)[:test_size//len(SNR_dB)//2].mean()
                loss_10 = (norm_test / torch.norm(h_test, dim=1)**2)[test_size-test_size//len(SNR_dB)//2:].mean()

                t_rec[itr-1] = tau_tbh_cplx[-1].mean().real
                lamb_rec[itr-1] = lamb.item()
            
            # pbar.set_description('tNMSE:%s, vNMSE:%s, t:%s, l:%s, g:%s' 
            #         %(format(float(iter_loss[itr-1]), '.3f'), format(float(testing_loss[itr-1]), '.3f'),
            #           format((t_rec[itr-1]), '5.2f'), format((lamb_rec[itr-1]), '.3f'), format((grad_norm(logits_net)), '.3f')))
            pbar.set_description('tNMSE:%s, vNMSE:%s, -4:%s, 10:%s, g:%s' 
                    %(format(float(iter_loss[itr-1]), '.3f'), format(float(testing_loss[itr-1]), '.3f'),
                      format(10*loss_n4.log10(), '.3f'), format(10*loss_10.log10(), '.3f'), format((grad_norm(logits_net)), '.2e')))
            pbar.update(1)

        priFeasibility = torch.mean(norm-c.T.matmul(tau_tbh_cplx.T).real)
        train_epochs = i+1
        if termination(logits_net, priFeasibility, lamb_rec[itr-1], epsilon):
            # train_epochs = i+1
            iter_loss = iter_loss[0:(i+1)*num_minibatch]#torch.zeros(num_epochs*num_minibatch).to(device)
            testing_loss = testing_loss[0:(i+1)*num_minibatch]
            t_rec = t_rec[0:(i+1)*num_minibatch]
            lamb_rec = lamb_rec[0:(i+1)*num_minibatch]
            break
            
    print('validation loss: ', iter_loss[itr-1].item())

    # Get the directory of the running script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lossPlot()
    parametersSave()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=1e-5) # learning rate
    parser.add_argument('-ep', type=int, default=10)    # num of epochs
    parser.add_argument('-tau', type=int, default=4)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('-nmb', type=int, default=10)  # 10 number of mini-batch
    parser.add_argument('-nR', type=int, default=4)  
    parser.add_argument('-nT', type=int, default=8)    
    parser.add_argument('-nI', type=int, default=8)  
    parser.add_argument('-T', type=int, default=8)
    parser.add_argument('-hsz', type=int, default=1, nargs='+')  # hidden layer size
    parser.add_argument('-cuda', type=int, default=0)  # cuda
    parser.add_argument('-snr', type=int, default=[-4,10], nargs='+')  
    parser.add_argument('-ch', type=str, default='default') 
    parser.add_argument('-psi', type=str, default='identity') 
    # parser.add_argument('-fsz', type=int, default=3)  # size of filters
    # parser.add_argument('-nc', type=int, default=8)  # num of channel of conv layer 
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available()) 

    torch.manual_seed(0)
    
    train([args.nT, args.nR, args.nI, args.T], args.hsz, args.lr, args.ep, 10**args.tau, args.nmb, args.cuda, args.snr, args.ch, args.psi)


    # print(2*4//2)
    # device = 'cuda:1'
    # SNR_lin = torch.tensor([0]).to(device)
    # h_c, y, h_mean, h_std = importData(2, n_R=2, n_I=2, n_T=2, T=2*4, SNR_lin=SNR_lin, device=device, W_Mean=0, IRScoef='h')
    # print(h_c.shape)

    # d,m,n = 3,4,2
    # a = torch.arange(d*m*n)
    # a = a.reshape(d,m,n)
    # print(a)
    # print(a.permute(0, 2, 1).reshape(d,m*n,1))
    # return