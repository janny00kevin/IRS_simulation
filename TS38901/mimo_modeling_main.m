function [H_channel] = mimo_modeling_main(AoD_mean_H, AoA_mean_H, AoD_AS_H, AoA_AS_H, kappa_H,nT,nR,dist_T2R,height_tx,height_rx,antenna_gain,delta_t,delta_r,M);
%
%%%%%%%%%%%%%%%%%%%%%%%%
% MIMO modeling program 
%%%%%%%%%%%%%%%%%%%%%%%%
%
% [H_channel H_a H_e Pi_H_a Pi_H_eig n_pc_a n_pc_eig] = mimo_modeling_main(AoD_mean_H, AoA_mean_H, AoD_AS_H, AoA_AS_H, kappa_H)
%
% outputs and displays the channel coefficient matrices H, H_a and H_e.
% The first coeff matrix uses angular domain representation (ADR), 
% described in
%
% D. Tse and P. Viswanath, \emph{Fundamental of Wireless Communication},
% Cambridge University Press, 2005. 
%
% to represent the MIMO channel. The second uses the eigenmodes of the
% one-sided correlation matrices described in
%
% W. Weichselberger \emph{et al.}, ``A stochastic MIMO channel model with
% joint correlation of both link ends,'' \emph{IEEE Trans. on Wireless 
% Communications}, vol. 5(1), pp. 90-100, Jan. 2006.
%
% 
% H_channel: contains MIMO channel elements (in spatial domain)
% H_a: contains the coefficients using the ADR
% H_e: contains the coefficients using the eigenmode
% Pi_H: indicator matrix indicating the location of the PCs in H_a which
%       retains 90% of the channel's energy
% n_pc: number of PCs
%
% AoD_mean_H: mean angle-of-departure of the channel, i.e. [0,360]
% AoA_mean_H: mean angle-of-arrival of the channel, i.e. [0,360]
% AoD_AS_H: angular spread at the transmitter, i.e. (0,360]
% AoA_AS_H: angular spread at the receiver, i.e.(0,360]
% kappa_H: strength of LOS components, e.g. 0-10 (can be more)
%

%clear all;

% Channel model parameters
if 0
    kappa_H     =   0;
    AoD_mean_H  =   (90/180)*pi;
    AoA_mean_H  =   (90/180)*pi;
    AoD_AS_H    =   (30/180)*pi;
    AoA_AS_H    =   (180/180)*pi;
end %0
AoD_mean_H(1) = (AoD_mean_H(1)/180)*pi;
AoA_mean_H(1) = (AoA_mean_H(1)/180)*pi;
AoD_AS_H(1) = (AoD_AS_H(1)/180)*pi;
AoA_AS_H(1) = (AoA_AS_H(1)/180)*pi;
% 
% AoD_mean_H(2) = (AoD_mean_H(2)/180)*pi;
% AoA_mean_H(2) = (AoA_mean_H(2)/180)*pi;
% AoD_AS_H(2) = (AoD_AS_H(2)/180)*pi;
% AoA_AS_H(2) = (AoA_AS_H(2)/180)*pi;

n_T         =   nT;
n_R         =   nR;
% total_num_ch = 2;
% 
% TSNR_dB     =   20;
% P_train     =   n_T;
% sigma_n_square  = P_train * (10^( - TSNR_dB/10 )) / (n_T*n_R);
% 
% N_buffer_1    =   1e+3;
% % N_buffer_2  =   1e+3;
% 
% Level_pc_th_a = 0.90;
% Level_pc_th_eig = 0.90;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Basis construction using ADR
% % Generate DFT Matrix of H @ Tx 2012/06/26
% for k_1 = 0:(n_T-1)
%     for k_2 = 0:(n_T-1)
%         U_T(k_1+1,k_2+1) = (1/sqrt(n_T)) * exp( -j*2*pi*k_1*k_2/n_T);
%     end
% end
% % Generate DFT Matrix of SU @ Rx 2012/06/26
% for k_1 = 0:(n_R-1)
%     for k_2 =0:(n_R-1)
%         U_R(k_1+1,k_2+1) = (1/sqrt(n_R)) * exp(-j*2*pi*k_1*k_2/n_R);
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% Generate Channel H
H_channel = zeros(n_R,n_T,1);
H_channel(:,:,1)   =   func_3gpp_scm_w_LOS(AoD_mean_H(1),AoA_mean_H(1),AoD_AS_H(1),AoA_AS_H(1),n_T,n_R,kappa_H(1),dist_T2R,height_tx,height_rx,antenna_gain,delta_t,delta_r,M);
% H_channel(:,:,2)   =   func_3gpp_scm_w_LOS(AoD_mean_H(2),AoA_mean_H(2),AoD_AS_H(2),AoA_AS_H(2),n_T,n_R,kappa_H(2),dist_T2R);
%%%%%%%%%%%%
% h_vec(:,1)   =   vec(H_channel(:,:,1));
% h_vec(:,2)   =   vec(H_channel(:,:,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting channel and channel coefficient matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Channel 1
% figure; bar3(1:size(H_channel(:,:,1),1), abs(H_channel(:,:,1))); title('|{\bf H}_1|');
% xlabel('Tx index'); ylabel('Rx index');
% title(['|{\bf H}_1|, ', '\kappa_{H_1} = ', num2str(kappa_H(1)), ', AoD_{mean,H_1} = ', num2str(AoD_mean_H(1)*180/pi), ', AoA_{mean,H_1} = ', num2str(AoA_mean_H(1)*180/pi), ', AoD_{AS,H_1} = ', num2str(AoD_AS_H(1)*180/pi), ', AoA_{AS,H_1} = ', num2str(AoA_AS_H(1)*180/pi)]);

%% Channel 2
% figure; bar3(1:size(H_channel(:,:,2),1), abs(H_channel(:,:,2))); title('|{\bf H}_2|');
% xlabel('Tx index'); ylabel('Rx index');
% title(['|{\bf H}_2|, ', '\kappa_{H_2} = ', num2str(kappa_H(2)), ', AoD_{mean,H_2} = ', num2str(AoD_mean_H(2)*180/pi), ', AoA_{mean,H_2} = ', num2str(AoA_mean_H(2)*180/pi), ', AoD_{AS,H_2} = ', num2str(AoD_AS_H(2)*180/pi), ', AoA_{AS,H_2} = ', num2str(AoA_AS_H(2)*180/pi)]);

%% the original function generate two channel but here we only need one.
% H_channel = H_channel(:,:,1);mimo_modeling_main([30,30], [30,30], [30,30], [30,30], [10,10], 10, 10, 100)

end

