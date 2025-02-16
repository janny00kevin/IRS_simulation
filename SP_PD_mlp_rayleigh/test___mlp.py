import torch
import torch.nn as nn
from torch.optim import Adam, SGD
# from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
# import time

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# def generateData(num_minibatch,num_trajectories, n_R, n_T, H_mean, H_sigma, SNR_lin, device, W_Mean):
#     sqrt2 = 2**0.5
#     ## generate communication data to train the parameterized policy
#     data_size = num_minibatch*num_trajectories
#     X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
#     H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(data_size, n_R, n_T, 2))/sqrt2).to(device)
#     S = H.matmul(X)
#     Ps = S.norm()**2/torch.ones_like(S.real).norm()**2
#     Pn = Ps / SNR_lin
#     Pn = Pn.repeat_interleave(num_minibatch*num_trajectories*n_R*T*2//len(SNR_lin)).reshape(num_minibatch*num_trajectories, n_R, T, 2)
#     W = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device)
#     Y = S + W
#     h = torch.view_as_real(H).reshape(data_size, n_R*n_T*2)
#     y = torch.view_as_real(Y).reshape(data_size, n_R*T*2)
#     return h, y

# def generateValData(test_size, n_R, n_T, H_mean, H_sigma, SNR_lin, device, W_Mean):
#     sqrt2 = 2**0.5
#     ## generate testing data
#     H_test = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(test_size, n_R, n_T, 2))/sqrt2).to(device)
#     Ps = H_test.norm()**2/torch.ones_like(H_test.real).norm()**2
#     Pn = Ps / SNR_lin
#     Pn = Pn.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(test_size//len(Pn), n_R, T, 2)
#     W_test = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device)
#     Y_test = H_test.matmul(torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)) + W_test   ###
#     h_test = torch.view_as_real(H_test).reshape(test_size, n_R*n_T*2)
#     y_test = torch.view_as_real(Y_test).reshape(test_size, n_R*T*2)
#     # h_test_norm = torch.norm((h_test).reshape(test_size, n_T*n_R*2), dim=1)**2
#     return h_test, y_test

def generateData(data_size, n_R, n_T, H_mean, H_sigma, SNR_lin, device, W_Mean):
    sqrt2 = 2**0.5
    ## generate communication data to train the parameterized policy
    # data_size = num_minibatch*num_trajectories
    X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    H = torch.view_as_complex(torch.normal(H_mean, H_sigma, size=(data_size, n_R, n_T, 2))/sqrt2).to(device)
    S = H.matmul(X)
    Ps = S.norm()**2/torch.ones_like(S.real).norm()**2
    Pn = Ps / SNR_lin
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R, T, 2)
    W = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device)
    Y = S + W
    h = torch.view_as_real(H).reshape(data_size, n_R*n_T*2)
    y = torch.view_as_real(Y).reshape(data_size, n_R*T*2)
    return h, y

def train(size, hidden_sizes=[64, 32], lr=1e-3, channel_information = [0, 1, 0, 0.1],
           num_epochs = 3, num_trajectories = 1e4, num_minibatch = 10, cuda = 1, snr = 0):

    def parametersSave():
    ## save the model and parameters
        checkpoint = {
            'logits_net': logits_net,
            'hidden_sizes': hidden_sizes,
            'num_epochs': num_epochs,
            'train_epochs': train_epochs,
            'num_minibatch': num_minibatch,
            'test_size': test_size,
            'n_R': n_R,
            'n_T': n_T,
            'T': T,
            'lr' : lr,
            'lr_l': lr_l,
            # 'lr_t': lr_t,
            'H_mean': H_mean,
            'H_sigma' :H_sigma,
            'W_Mean': W_Mean,
            'SNR_dB': SNR_dB,
            't_rec': t_rec,
            # 'mse': lse,
            'lamb_rec': lamb_rec,
            'iter_loss_N': iter_loss_N.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"),
            'testing_loss_N': testing_loss_N.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"),         
        }
        torch.save(checkpoint, './simulation/result/SP_lr%s_%s_ep%s_SNR%s.pt' %(lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T+2],train_epochs,SNR_dB.tolist()))
    
    def NMSEplot(epochs):
        plt.subplot(311)
        plt.plot(epochs, iter_loss_N.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training nmse')
        plt.plot(epochs, testing_loss_N.reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation nmse')
        plt.suptitle("MMSE based with sample avg PD channel estimator")
        plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, SNR:%s~%s' 
                  %(n_R,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T+2], torch.min(SNR_dB).item(), torch.max(SNR_dB).item()))
        
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

        plt.savefig('./simulation/result/SP_lr%s_%s_ep%s_SNR_%s.png' 
                    %(lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T+2],train_epochs,SNR_dB.tolist()))
        plt.close()

    # def shuffle():
    #     return
    def grad_norm(logits_net):
        gradient_norm = 0
        for _, param in logits_net.named_parameters():
            # print(f"Gradient of {name}: {param.grad}")
            gradient_norm += torch.norm(param.grad)**2
        return gradient_norm
    
    def termination(logits_net, priFeasibility, lambda_val, epsilon):
        # calculate_norm(y, X_tilde, D, h_tilde) < epsilon or k == num_iterations-1
        gradient_norm = grad_norm(logits_net)
        return gradient_norm <= epsilon and priFeasibility <= 0 and lambda_val.real >= 0 and lambda_val.real*priFeasibility <= epsilon

    device = torch.device("cuda:%s"%(cuda) if torch.cuda.is_available() else "cpu")
    n_T, n_R, T = size
    # data_size = num_minibatch*num_trajectories
    H_mean, H_sigma, W_Mean = channel_information
    test_size = 2000                                                       
    SNR_dB = torch.tensor(list(range(min(snr),max(snr)+1,2))).to(device)   ###
    SNR_lin = 10**(SNR_dB/10.0)
    print('SNR_dB:',SNR_dB)
    
    ## generate train and test data
    h, y = generateData(num_minibatch*num_trajectories, n_R, n_T, H_mean, H_sigma, SNR_lin, device, W_Mean)
    h_test, y_test = generateData(test_size, n_R, n_T, H_mean, H_sigma, SNR_lin, device, W_Mean)
    # X = torch.complex(torch.eye(n_T).tile(T//n_T), torch.zeros(n_T, T)).to(device)
    # XT = torch.complex(torch.eye(n_T).tile(T//n_T,1), torch.zeros(T, n_T)).to(device)
    # X_tilde = torch.kron(XT, torch.eye(n_R).to(device))
    c = torch.cat((torch.zeros(n_R*n_T,1), torch.ones(1,1))).to(torch.complex64).to(device)
    D = torch.cat((torch.eye(n_R*n_T), torch.zeros(n_R*n_T,1)),1).to(torch.complex64).to(device)
    
    ## initialize t, lambda   
    lamb = torch.tensor([1.0], requires_grad=True, device=device)
    lr_l = 1e-4 #5e-5                                                    ###
    epsilon = 1e-2
    
    ## generate MLP ## initialize \tbh
    logits_net = mlp(sizes=[2*n_R*T]+hidden_sizes+[2*n_R*n_T+2]).to(device)
    print(logits_net)
    optimizer_pri = Adam(logits_net.parameters(), lr=lr)
    # optimizer_l = Adam([lamb], lr=lr_l, betas=(0.5, 0.999))

    iter_loss_N = torch.zeros(num_epochs*num_minibatch).to(device)
    testing_loss_N = torch.zeros(num_epochs*num_minibatch).to(device)
    t_rec = torch.zeros(num_epochs*num_minibatch).to(device)
    lamb_rec = torch.zeros(num_epochs*num_minibatch).to(device)
    train_epochs = 0
    # lse = torch.zeros(num_epochs*num_minibatch).to(device)

    ## plot progress bar
    pbar = tqdm(total = num_epochs*num_minibatch)
    for i in range(num_epochs):
        torch.cuda.empty_cache()
        ### after 1 epoch, shuffle the dataset, using the same index to shuffle H and Y
        # y = y.reshape(num_minibatch, num_trajectories*n_R*T*2)
        # h = h.reshape(num_minibatch, num_trajectories*n_R*T*2)
        # idx = torch.randperm(num_minibatch).to(device)
        # y = y.index_select(0, idx)
        # h = h.index_select(0, idx)
        # y = y.reshape(num_minibatch*num_trajectories, n_R*T*2)
        # h = h.reshape(num_minibatch*num_trajectories, n_R*T*2)

        idx = torch.randperm(num_minibatch*num_trajectories).to(device)
        y = y[idx,:]
        h = h[idx,:]
        for j in range(num_minibatch):
            itr = j+i*num_minibatch+1
            ### trajectories training data
            tau_y = y[j*num_trajectories:(j+1)*num_trajectories, :]
            tau_h = h[j*num_trajectories:(j+1)*num_trajectories, :]
            ### feed into the NN
            logits = logits_net(tau_y)
            tau_tbh_cplx = torch.view_as_complex(logits.reshape(num_trajectories, n_T*n_R+1, 2))
            # tau_h_hat = Normal(logits[:,::2], torch.exp(logits[:,1::2])).sample()
            
            ### compute loss and update
            optimizer_pri.zero_grad()
            ## 2-norm square of y - \tbX\D\phi at each realization (size: num_trajectories by 1) 
            norm = torch.norm(torch.view_as_complex((tau_h).reshape(num_trajectories, n_R*n_T, 2)) - D.matmul(tau_tbh_cplx.T).T, dim=1)**2

            lamb_dt = lamb.detach().clone()

            loss_pri = ((1-lamb_dt)*c.T.matmul(tau_tbh_cplx.T).real + lamb_dt*norm).mean()
            loss_pri.backward()
            optimizer_pri.step()
            
            ## lambda update
            lamb = lamb + lr_l/itr * torch.mean(norm-c.T.matmul(tau_tbh_cplx.T).real)
            if lamb.item() < 0:
                lamb.data.fill_(0)

            with torch.no_grad():
                ## save the loss to plot
                # norm2 = torch.norm(torch.view_as_complex(tau_h.reshape(num_trajectories, n_T*n_R, 2)) - D.matmul(tau_tbh_cplx.T).T, dim=1)**2
                iter_loss_N[itr-1] = (norm / torch.norm((tau_h).reshape(num_trajectories, n_T*n_R*2), dim=1)**2).mean()
                # iter_loss_N[itr-1] = norm.mean()
                
                ## validation
                logits_test = logits_net(y_test)
                test_tbh_cplx = torch.view_as_complex(logits_test.reshape(test_size, n_T*n_R+1, 2))
                # norm_test = torch.norm(torch.view_as_complex(h_test.reshape(test_size, n_T*n_R, 2)) - D.matmul(test_tbh_cplx.T).T, dim=1)**2
                norm_test = torch.norm(torch.view_as_complex((h_test).reshape(test_size, n_R*T, 2)) - D.matmul(test_tbh_cplx.T).T, dim=1)**2
                testing_loss_N[itr-1] = (norm_test / torch.norm((h_test).reshape(test_size, n_T*n_R*2), dim=1)**2).mean()
                loss_n4 = (norm_test / torch.norm(h_test, dim=1)**2)[:test_size//len(SNR_dB)//2].mean()
                loss_10 = (norm_test / torch.norm(h_test, dim=1)**2)[test_size-test_size//len(SNR_dB)//2:].mean()
                # testing_loss_N[itr-1] = norm_test.mean()

                t_rec[itr-1] = tau_tbh_cplx[-1].mean().real
                lamb_rec[itr-1] = lamb.item()
                # lse[itr-1] = norm.mean()
                # if itr == 1:
                #     # print(logits.grad)
                #     for name, param in logits_net.named_parameters():
                #         print(f"Gradient of {name}: {param.grad}")
            
            # pbar.set_description('tNMSE:%s, vNMSE:%s, t:%s, l:%s, g:%s' 
            #         %(format(float(iter_loss_N[itr-1]), '.3f'), format(float(testing_loss_N[itr-1]), '.3f'),
            #           format((t_rec[itr-1]), '5.2f'), format((lamb_rec[itr-1]), '.3f'), format((grad_norm(logits_net)), '.3f')))
            pbar.set_description('tNMSE:%s, vNMSE:%s, -4:%s, 10:%s, g:%s' 
                    %(format(float(iter_loss_N[itr-1]), '.3f'), format(float(testing_loss_N[itr-1]), '.3f'),
                      format(loss_n4, '.3f'), format(loss_10, '.3f'), format((grad_norm(logits_net)), '.2e')))
            pbar.update(1)
            # time.sleep(0.1)

        priFeasibility = torch.mean(norm-c.T.matmul(tau_tbh_cplx.T).real)
        train_epochs = i+1
        if termination(logits_net, priFeasibility, lamb_rec[itr-1], epsilon):
            # train_epochs = i+1
            iter_loss_N = iter_loss_N[0:(i+1)*num_minibatch]#torch.zeros(num_epochs*num_minibatch).to(device)
            testing_loss_N = testing_loss_N[0:(i+1)*num_minibatch]
            t_rec = t_rec[0:(i+1)*num_minibatch]
            lamb_rec = lamb_rec[0:(i+1)*num_minibatch]
            break
            
    lossPlot()
    parametersSave()

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--hs', type=float, default=1) # sigma of H
    # parser.add_argument('--hm', type=float, default=0) # mean of H
    parser.add_argument('--lr', type=float, default=1e-5) # learning rate
    parser.add_argument('--ep', type=int, default=10)    # num of epochs
    parser.add_argument('--tau', type=int, default=4)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('--nmb', type=int, default=100)  # 100 number of mini-batch
    parser.add_argument('--nR', type=int, default=4)  
    parser.add_argument('--nT', type=int, default=8)  
    parser.add_argument('--hsz', type=int, default=1, nargs='+')  # hidden layer size
    parser.add_argument('--cuda', type=int, default=0)  # cuda
    parser.add_argument('--snr', type=int, default=[-4,10], nargs='+')  
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available())
    # print(args.hsz)

    # if args.hsz == 1:
    #     hidden_sizes = [64, 32]
    # else:
    #     hidden_sizes = args.hsz

    T = args.nT*1
    channel_information = [0, 1, 0]

    train([args.nT, args.nR, T], args.hsz, args.lr, channel_information, args.ep, 10**args.tau, args.nmb, args.cuda, args.snr)
