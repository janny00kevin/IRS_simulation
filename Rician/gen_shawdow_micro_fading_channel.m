function H_rician = gen_shawdow_micro_fading_channel(nTx, nRx, std_xi_dB_0, antenna_gain,K_db)
%{

This function aims to generate a white channel that only concerning micro
fading and shadowing effect in macro-fading !!!

%}
%% parameters
% K_db = 10; %(dB)
K_linear = 10*log(K_db/10);

% Rician fading distribution from non-zero mean Gaussian
mean = sqrt(K_linear/(2*(K_linear + 1)));
standard_deviation = sqrt(1/(2*(K_linear + 1)));

% height of Transmitter and Receiver
% h_T = h_Tx;
% h_R = h_Rx;

fc = 28; %(GHz) carrier frequency

% dist_3D = sqrt( (h_T-h_R)^2 + (dist_Tx2Rx)^2);

%% gen channel here

H_w_rician =  ( (mean + standard_deviation*randn(nRx, nTx)) + 1j*(mean + standard_deviation*randn(nRx, nTx)))/sqrt(2);
% H_w_NLOS = (randn(nRx, nTx) + 1j*randn(nRx, nTx))/sqrt(2);

xi_LOS = std_xi_dB_0*randn(nTx,1); % random shadowing

PL_0 = 32.4; % (dB)
PL_LOS = PL_0 + 20*log10(fc); % dB

betta_LOS = ones(nTx,1).*10^(antenna_gain/10).*10.^(xi_LOS/10).*10^(-PL_LOS/10);


H_rician  = H_w_rician*diag(sqrt(betta_LOS)); % combine micro and macro fading


end