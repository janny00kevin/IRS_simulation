import torch
from torch.optim import Adam
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_path)))
from utils.p2p_ct_channels import import_data
from utils.complex_utils import turn_real, turn_cplx
from utils.NN_model.AE_MLP_1024 import SparseAutoencoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(size, hidden_sizes, lr, num_epochs, batch_size, num_minibatch, cuda, snr, channel, eps_fidel):
    
    def parameters_save():
    ## save the model and parameters
        checkpoint = {
            'logits_net': best_model,
            'num_epochs': num_epochs, 'train_epochs': train_epochs, 'num_minibatch': num_minibatch, 'test_size': test_size,
            'n_R': n_R, 'n_T': n_T, 'T': T, 'lr' : lr, 'SNR_dB': SNR_dB,
            'iter_loss': iter_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"),
            'testing_loss': testing_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), 
            'h_mean': h_mean, 'h_std': h_std,
        }
        torch.save(checkpoint, save_path + '.pt')
    
    def NMSE_plot(epochs):
        plt.subplot(311)
        plt.plot(epochs, iter_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training nmse')
        plt.plot(epochs, testing_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation nmse')
        plt.suptitle("parametric PD trained AE MIMO ChEst" )
        plt.title(' $[n_R,n_T,T]$:[%s,%s,%s], lr:%s, size:%s, SNR:%s~%s' 
                  %(n_R,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T], torch.min(SNR_dB).item(), torch.max(SNR_dB).item()))
        
        plt.xlabel('epochs')
        plt.legend()
        plt.grid(True)

    def t_MSE_plot(epochs):
        return
        plt.subplot(312)
        plt.plot(epochs,t_rec[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='t')
        plt.xlabel('epochs')
        plt.grid(True)
        plt.legend()

    def lambda_plot(epochs):
        plt.subplot(313)
        plt.plot(epochs,lamb_rec[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"),label='$\\lambda$')
        plt.xlabel('epochs')
        plt.grid(True)
        plt.legend()
    
    def loss_plot():
        epochs = range(1, train_epochs+1)
        NMSE_plot(epochs)
        t_MSE_plot(epochs)
        lambda_plot(epochs)

        plt.savefig(save_path + '.pdf')
        plt.close()

    def grad_norm(logits_net):
        gradient_norm = 0
        for _, param in logits_net.named_parameters():  gradient_norm += torch.norm(param.grad)**2
        return gradient_norm

    def KKT_termination(logits_net, constraint_loss, lambda_val, epsilon):
        gradient_norm = grad_norm(logits_net)
        return gradient_norm/num_params <= epsilon and constraint_loss <= 0 and\
               lambda_val >= 0 and lambda_val*constraint_loss <= epsilon

    device = torch.device("cuda:%s"%(cuda) if torch.cuda.is_available() else "cpu")
    n_T, _, n_R, T = size
    train_size = num_minibatch*batch_size
    test_size = 2000
    SNR_dB = torch.tensor(list(range(min(snr),max(snr)+1,2))).to(device)   ###
    SNR_lin = 10**(SNR_dB/10.0)
    eps_fidel = torch.tensor(eps_fidel, dtype=torch.float32).to(device)  
    print('training with SNR_dB:',SNR_dB)

    ## load training and testing data
    h, y, h_mean, h_std = import_data(train_size, n_R, n_T, T, SNR_lin, device, phase = 'train', channel=channel)
    h_test, y_test, _, _ = import_data(test_size, n_R, n_T, T, SNR_lin, device, phase = 'val',   channel=channel)
    print('h_mean:',h_mean, 'h_std:',h_std)
    # c = torch.cat((torch.zeros(n_R*n_T*n_I,1), torch.ones(1,1))).to(torch.complex64).to(device)
    # D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

    ## initialize lambda   
    lamb = torch.tensor([0.0], requires_grad=True, device=device)    ####################################################
    lr_l = 1e-4 #5e-5                                             
    KKT_thres = 1e-2

    ## generate MLP ## initialize \tbh
    logits_net = SparseAutoencoder(2*n_R*T, hidden_sizes[0], 2*n_R*n_T).to(device)
    # print(logits_net)
    num_params = sum(p.numel() for p in logits_net.parameters() if p.requires_grad)
    optimizer_pri = Adam(logits_net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer_pri, mode='min', patience=10, factor=0.5, threshold=1e-7)  ####################################################
    current_lr = optimizer_pri.param_groups[0]['lr']

    ## initialize the training record
    num_iters = num_epochs*num_minibatch
    iter_loss = torch.zeros(num_iters).to(device)
    testing_loss = torch.zeros(num_iters).to(device)
    # t_rec = torch.zeros(num_iters).to(device)
    lamb_rec = torch.zeros(num_iters).to(device)
    train_epochs = 0

    ## Early stopping variables
    best_loss = float('inf')  # Initialize the best loss as infinity
    min_lr = 5e-7  # Minimum learning rate
    best_model = None  # Initialize the best model as None
    lr_update_count = 0  # Initialize the learning rate update count
    best_n4 = float('inf')  # Initialize the best n4 loss as infinity
    best_10 = float('inf')  # Initialize the best 10 loss as infinity

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
            tau_y = y[j*batch_size:(j+1)*batch_size, :]
            tau_h = h[j*batch_size:(j+1)*batch_size, :]

            ### feed into the NN
            logits_net.train()
            logits, latent = logits_net(tau_y)
            latent_cplx = turn_cplx(latent)
            # logits_cplx = turn_cplx(logits)
            
            ### compute loss and update
            optimizer_pri.zero_grad()
            l1_loss = torch.mean(torch.norm(latent_cplx, p=1, dim=1))
            constraint_loss = torch.mean(torch.norm(tau_h - logits, p=2, dim=1)**2) - eps_fidel
            # norm = torch.norm(turnCplx(tau_h) - D.matmul(tau_tbh_cplx.T).T, dim=1)**2

            ## dual variable lambda update
            with torch.no_grad():
                lamb = lamb + lr_l*torch.exp(-0.0001*torch.tensor(itr)) * constraint_loss.detach()  ####################################################
                lamb[lamb < 0] = 0

            # lamb_dt = lamb.detach().clone()

            ## primal variable theta update
            lagrangian = l1_loss + lamb*(constraint_loss)
            # loss_pri = ((1-lamb_dt)*c.T.matmul(tau_tbh_cplx.T).real + lamb_dt*norm).mean()
            lagrangian.backward()
            optimizer_pri.step()
            
            with torch.no_grad():
                ## save the loss to plot
                iter_loss[itr-1] = (torch.norm(tau_h - logits, dim=1)**2 / torch.norm(tau_h, dim=1)**2).mean()
                
                ## validation
                logits_net.eval()
                logits_test = turn_cplx(logits_net(turn_real(turn_cplx(y_test)-h_mean)/h_std)[0])*h_std + h_mean
                # test_tbh_cplx = turn_cplx(logits_test)*h_std + h_mean
                nmlz_error_test = torch.norm(turn_cplx(h_test) - logits_test, dim=1)**2 / torch.norm(h_test, dim=1)**2
                testing_loss[itr-1] = nmlz_error_test.mean()
                loss_n4 = nmlz_error_test[:test_size//len(SNR_dB)].mean()
                loss_10 = nmlz_error_test[test_size-test_size//len(SNR_dB):].mean()

                # t_rec[itr-1] = tau_tbh_cplx[-1].mean().real
                lamb_rec[itr-1] = lamb.item()

                # if (testing_loss[itr-1] < best_loss): # and lr_update_count > 0 
                    
                #     best_loss = testing_loss[itr-1].item()
                #     best_n4 = loss_n4.item()
                #     best_10 = loss_10.item()
                #     best_model = logits_net
                    
                # if KKT_termination(logits_net, constraint_loss, lamb_rec[itr-1], epsilon) or 0:
                #     iter_loss = iter_loss[0:(i+1)*num_minibatch]#torch.zeros(num_epochs*num_minibatch).to(device)
                #     testing_loss = testing_loss[0:(i+1)*num_minibatch]
                #     # t_rec = t_rec[0:(i+1)*num_minibatch]
                #     lamb_rec = lamb_rec[0:(i+1)*num_minibatch]
                #     print(f"Early stopping at epoch {i+1}, iteration {j+1}. KKT conditions met.")
                #     best_loss = testing_loss[itr-1].item()
                #     best_n4 = loss_n4.item()
                #     best_10 = loss_10.item()
                #     best_model = logits_net
                #     break
            
            pbar.set_description('tNMSE:%s, vNMSE:%s, -4:%s, 10:%s, l:%s, g:%s, f:%s' 
                    %(format(float(iter_loss[itr-1]), '.3f'), format(float(testing_loss[itr-1]), '.3f'),
                      format(10*loss_n4.log10(), '.3f'), format(10*loss_10.log10(), '.3f'), format(lamb_rec[itr-1], '.2f'),
                      format(grad_norm(logits_net).item()/num_params, '.2e'), format(constraint_loss.item(), '.2f')))
            # pbar.set_description(f'tNMSE:{iter_loss[itr-1]:.3f}, vNMSE:{testing_loss[itr-1]:.3f},\n'
            #          f'-4:{10 * loss_n4.log10():.3f}, 10:{10 * loss_10.log10():.3f},\n'
            #          f'l:{lamb_rec[itr-1]:.2f}')
            pbar.update(1)

        ### early stopping
        train_epochs = i+1
        scheduler.step(testing_loss[i*num_minibatch:(i+1)*num_minibatch].mean())
        if current_lr != optimizer_pri.param_groups[0]['lr']:
            current_lr = optimizer_pri.param_groups[0]['lr']
            # print(f"Epoch {i+1}: Learning rate is {current_lr:.1e},  best loss: {best_loss:.3f}, best n4: {best_n4:.3f}, best 10: {best_10:.3f}")
            print(f"Epoch {i+1}: Learning rate is {current_lr:.1e}")
            lr_update_count = lr_update_count + 1     ##############
        if current_lr < min_lr and KKT_termination(logits_net, constraint_loss, lamb_rec[itr-1], KKT_thres):
            iter_loss = iter_loss[0:(i+1)*num_minibatch]#torch.zeros(num_epochs*num_minibatch).to(device)
            testing_loss = testing_loss[0:(i+1)*num_minibatch]
            # t_rec = t_rec[0:(i+1)*num_minibatch]
            lamb_rec = lamb_rec[0:(i+1)*num_minibatch]
            print(f"Stopping training as learning rate reached {current_lr:.2e} at epoch {i+1}, KKT conditions met.")
            best_loss = testing_loss[itr-1].item()
            best_n4 = loss_n4.item()
            best_10 = loss_10.item()
            best_model = logits_net
            break

        # primal_feasibility = torch.mean(norm-c.T.matmul(tau_tbh_cplx.T).real)
        # if termination(logits_net, constraint_loss, lamb_rec[itr-1], epsilon) or 0:
        #     iter_loss = iter_loss[0:(i+1)*num_minibatch]#torch.zeros(num_epochs*num_minibatch).to(device)
        #     testing_loss = testing_loss[0:(i+1)*num_minibatch]
        #     # t_rec = t_rec[0:(i+1)*num_minibatch]
        #     lamb_rec = lamb_rec[0:(i+1)*num_minibatch]
        #     break
            
    print(f"Training completed after {train_epochs} epochs. Best loss: {best_loss:.3f}, best n4: {best_n4:.3f}, best 10: {best_10:.3f}")
    pbar.close()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'trained_model', '%.3f_eps%.1f_%s_AE_lr%.0e_%s_ep%s' 
            %(best_loss, eps_fidel, channel, lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T], train_epochs))
    loss_plot()
    parameters_save()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=1e-3) # learning rate
    parser.add_argument('-ep', type=int, default=1000)    # num of epochs
    parser.add_argument('-mbs', type=int, default=900)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('-nmb', type=int, default=100)  # 10 number of mini-batch
    parser.add_argument('-nT', type=int, default=36) 
    parser.add_argument('-nI', type=int, default=0)
    parser.add_argument('-nR', type=int, default=4)       
    parser.add_argument('-T', type=int, default=36)
    parser.add_argument('-hsz', type=int, default=[1024], nargs='+')  # hidden layer size
    parser.add_argument('-cuda', type=int, default=0)  # cuda
    parser.add_argument('-snr', type=int, default=[-4,10], nargs='+')  
    parser.add_argument('-ch', type=str, default='default') 
    parser.add_argument('-eps', type=float, default=1e1)  # 5e1    ####################################################
    # parser.add_argument('-psi', type=str, default='h')
    # parser.add_argument('-lyr', type=int, default=0)
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available()) 

    torch.manual_seed(0)
    assert args.mbs*args.nmb == 9e4, 'mbs*nmb should be 9e4'

    import sys
    print("Command line arguments:", sys.argv)
    
    train([args.nT, args.nI, args.nR, args.T], args.hsz, args.lr, args.ep, args.mbs, args.nmb, args.cuda, args.snr, args.ch, args.eps)

