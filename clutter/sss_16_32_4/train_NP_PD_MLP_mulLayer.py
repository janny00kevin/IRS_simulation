import torch
from torch.optim import Adam
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from utils.IRS_ct_channels_8snr import import_data
from utils.complex_utils import turn_real, turn_cplx, vec
# from utils.NN_model.MLP_mulLayer_1024 import MLP
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(size, hidden_sizes, lr, num_epochs, batch_size, num_minibatch, cuda, snr, channel, IRS_coe_type, mlp_layer):
    
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
        plt.plot(epochs, iter_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='training nmse')
        plt.plot(epochs, testing_loss[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation nmse')
        plt.suptitle("nonparametric PD trained MMSE MIMO ChEst with IRS coefficient %s" %IRS_coe_type)
        plt.title(' $[n_R,n_I,n_T,T]$:[%s,%s,%s,%s], lr:%s, size:%s, SNR:%s~%s' 
                  %(n_R,n_I,n_T,T,lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T*n_I+2], torch.min(SNR_dB).item(), torch.max(SNR_dB).item()))
        
        plt.xlabel('epochs')
        plt.legend()
        plt.grid(True)

    def t_MSE_Plot(epochs):
        plt.subplot(312)
        plt.plot(epochs,t_rec[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"), label='t')
        plt.xlabel('epochs')
        plt.grid(True)
        plt.legend()

    def lambPlot(epochs):
        plt.subplot(313)
        plt.plot(epochs,lamb_rec[:train_epochs*num_minibatch].reshape(train_epochs,num_minibatch).mean(dim=1).to("cpu"),label='$\\lambda$')
        plt.xlabel('epochs')
        plt.grid(True)
        plt.legend()
    
    def lossPlot():
        ## plot the loss
        epochs = range(1, train_epochs+1)
        NMSEplot(epochs)
        t_MSE_Plot(epochs)
        lambPlot(epochs)

        plt.savefig(save_path + '.pdf')
        plt.close()

    def grad_norm(logits_net):
        gradient_norm = 0
        for _, param in logits_net.named_parameters():  gradient_norm += torch.norm(param.grad)**2
        return gradient_norm
    
    def KKT_termination(logits_net, priFeasibility, lambda_val, epsilon):
        # calculate_norm(y, X_tilde, D, h_tilde) < epsilon or k == num_iterations-1
        gradient_norm = grad_norm(logits_net)
        return gradient_norm/num_params <= epsilon and priFeasibility <= 0 and\
                lambda_val.real >= 0 and lambda_val.real*priFeasibility <= epsilon
    
    if mlp_layer == 2:
        from utils.NN_model.MLP_2layer_1024 import MLP
    elif mlp_layer == 5:
        from utils.NN_model.MLP_5Layer_1024 import MLP
    elif mlp_layer == 10:
        from utils.NN_model.MLP_10Layer_1024 import MLP
    elif mlp_layer == 20:
        from utils.NN_model.MLP_20Layer_1024 import MLP
    else:
        assert False, 'mlp_layer (-lyr) should be 2, 5, 10 or 20'

    device = torch.device("cuda:%s"%(cuda) if torch.cuda.is_available() else "cpu")
    n_T, n_I, n_R, T = size
    train_size = num_minibatch*batch_size
    test_size = 2000
    SNR_dB = torch.tensor(list(range(min(snr),max(snr)+1,2))).to(device)   ###
    SNR_lin = 10**(SNR_dB/10.0)
    print('training with SNR_dB:',SNR_dB)

    ## load training and testing data
    h, y, h_mean, h_std = import_data(train_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type, phase = 'train', channel=channel)
    h_test, y_test, _, _ = import_data(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=IRS_coe_type, phase = 'val', channel=channel)
    # print('h_mean:', h_mean, 'h_std:', h_std)
    c = torch.cat((torch.zeros(n_R*n_T*n_I,1), torch.ones(1,1))).to(torch.complex64).to(device)
    D = torch.cat((torch.eye(n_R*n_T*n_I), torch.zeros(n_R*n_T*n_I,1)),1).to(torch.complex64).to(device)

    ## initialize lambda   
    lamb = torch.tensor([1.0], requires_grad=True, device=device)
    lr_l = 1e-4 #5e-5                                                    ###
    KKT_thres = 3e-1

    ## generate MLP ## initialize \tbh
    logits_net = MLP(2*n_R*T, hidden_sizes[0], 2*n_R*n_T*n_I+2).to(device)
    num_params = sum(p.numel() for p in logits_net.parameters() if p.requires_grad)
    # print(logits_net)
    optimizer_pri = Adam(logits_net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer_pri, mode='min', patience=3, factor=0.5, threshold=1e-3)
    current_lr = optimizer_pri.param_groups[0]['lr']

    ## initialize the training record
    num_iters = num_epochs*num_minibatch
    iter_loss = torch.zeros(num_iters).to(device)
    testing_loss = torch.zeros(num_iters).to(device)
    t_rec = torch.zeros(num_iters).to(device)
    lamb_rec = torch.zeros(num_iters).to(device)
    train_epochs = 0

    # Early stopping
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
            logits = logits_net(tau_y)
            tau_tbh_cplx = turn_cplx(logits)
            
            ### compute loss and update
            optimizer_pri.zero_grad()
            ## 2-norm square of y - \tbX\D\phi at each realization (size: batch_size by 1) 
            norm = torch.norm(turn_cplx(tau_h) - D.matmul(tau_tbh_cplx.T).T, dim=1)**2

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
                pri_feasibility = torch.mean(norm-c.T.matmul(tau_tbh_cplx.T).real)

                ## save the loss to plot
                iter_loss[itr-1] = (norm / torch.norm(tau_h, dim=1)**2).mean()
                
                ## validation
                logits_net.eval()
                logits_test = logits_net(turn_real(turn_cplx(y_test)-h_mean)/h_std)
                test_tbh_cplx = turn_cplx(logits_test)*h_std + h_mean
                # logits_test = logits_net(y_test)
                # test_tbh_cplx = turnCplx(logits_test)
                norm_test = torch.norm(turn_cplx(h_test) - (D.matmul(test_tbh_cplx.T).T), dim=1)**2
                testing_loss[itr-1] = (norm_test / torch.norm(h_test, dim=1)**2).mean()
                loss_n4 = (norm_test / torch.norm(h_test, dim=1)**2)[:test_size//len(SNR_dB)//2].mean()
                loss_10 = (norm_test / torch.norm(h_test, dim=1)**2)[test_size-test_size//len(SNR_dB)//2:].mean()

                t_rec[itr-1] = tau_tbh_cplx[-1].mean().real
                lamb_rec[itr-1] = lamb.item()

                if current_lr < lr and testing_loss[itr-1] < best_loss and KKT_termination(logits_net, pri_feasibility, lamb_rec[itr-1], KKT_thres):
                    best_loss = testing_loss[itr-1].item()
                    best_n4 = loss_n4.item()
                    best_10 = loss_10.item()
                    # best_model = logits_net
            
            # pbar.set_description('tNMSE:%s, vNMSE:%s, t:%s, l:%s, g:%s' 
            #         %(format(float(iter_loss[itr-1]), '.3f'), format(float(testing_loss[itr-1]), '.3f'),
            #           format((t_rec[itr-1]), '5.2f'), format((lamb_rec[itr-1]), '.3f'), format((grad_norm(logits_net)), '.3f')))
            pbar.set_description('tNMSE:%s, vNMSE:%s, -4:%s, 10:%s, g:%s' 
                    %(format(float(iter_loss[itr-1]), '.3f'), format(float(testing_loss[itr-1]), '.3f'),
                      format(10*loss_n4.log10(), '.3f'), format(10*loss_10.log10(), '.3f'), 
                      format((grad_norm(logits_net).item()/num_params), '.2e')))
            pbar.update(1)

        ### early stopping
        train_epochs = i+1
        scheduler.step(testing_loss[i*num_minibatch:(i+1)*num_minibatch].mean())
        if current_lr != optimizer_pri.param_groups[0]['lr']:
            current_lr = optimizer_pri.param_groups[0]['lr']
            print(f"Epoch {i+1}: Learning rate is {current_lr:.1e},  best loss: {best_loss:.3f}, best n4: {best_n4:.3f}, best 10: {best_10:.3f}")
            lr_update_count = lr_update_count + 1     ##############
        if current_lr < min_lr and KKT_termination(logits_net, pri_feasibility, lamb_rec[itr-1], KKT_thres):
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
            
    print(f"Training completed after {train_epochs} epochs. Best loss: {best_loss:.3f}, best n4: {best_n4:.3f}, best 10: {best_10:.3f}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'trained_model', '%.3f_8SNR_%s_%sMLP_psi_%s_lr%.0e_%s_ep%s' 
            %(best_loss, channel, mlp_layer, IRS_coe_type, lr, [2*n_R*T]+hidden_sizes+[2*n_R*n_T*n_I+2], train_epochs))
    lossPlot()
    parametersSave()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=1e-3) # learning rate
    parser.add_argument('-ep', type=int, default=1000)    # num of epochs
    parser.add_argument('-mbs', type=int, default=1000)  # 10k length of trajectory = size of mini-batch
    parser.add_argument('-nmb', type=int, default=100)  # 10 number of mini-batch
    parser.add_argument('-nT', type=int, default=16) 
    parser.add_argument('-nI', type=int, default=32)
    parser.add_argument('-nR', type=int, default=4)       
    parser.add_argument('-T', type=int, default=512)
    parser.add_argument('-hsz', type=int, default=[1024], nargs='+')  # hidden layer size
    parser.add_argument('-cuda', type=int, default=0)  # cuda
    parser.add_argument('-snr', type=int, default=[-4,10], nargs='+')  
    parser.add_argument('-ch', type=str, default='default') 
    parser.add_argument('-psi', type=str, default='h')
    parser.add_argument('-lyr', type=int, default=0)
    args = parser.parse_args()

    print("cuda:%s"%(args.cuda), torch.cuda.is_available()) 

    torch.manual_seed(0)
    assert args.mbs*args.nmb == 1e5, 'mbs*nmb should be 1e6'
    
    train([args.nT, args.nI, args.nR, args.T], args.hsz, args.lr, args.ep, args.mbs, args.nmb, args.cuda, args.snr, args.ch, args.psi, args.lyr)

