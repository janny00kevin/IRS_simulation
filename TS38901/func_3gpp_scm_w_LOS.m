function H_channel = func_3gpp_scm_w_LOS(AoD_mean_H,AoA_mean_H,AoD_AS_H,AoA_AS_H,n_T,n_R,kappa_H,dist_T2R,height_tx,height_rx,antenna_gain,delta_t,delta_r,M)
% delta: inter antenna distance
% M: number of pathes(NLOS)
%% including of macro fading part considering path_loss and shawdowing (i.e. considering the distance as a factor)
c = 3e8;
h_Tx = height_tx; % the height of Transmiter
h_Rx = height_rx; % the height of Receiver
fc = 28;  % carrier frequency, 28GHz
lamba_c = 3e8/(fc*1e9); % wavelength
deltan_t = delta_t/lamba_c; %normalized interantenna distance
deltan_r = delta_r/lamba_c; %normalized interantenna distance
D3d = sqrt(( h_Tx - h_Rx )^2 + dist_T2R^2  );  % calculate the distacne in 3D (dist_T2R is horizontal distance)

std_xi_LOS = 4;      % standard deviation of shadowing factor
std_xi_NLOS = 7.82;    % standard deviation of shadowing factor
% std_xi_LOS = 0;      % standard deviation of shadowing factor
% std_xi_NLOS = 0;    % standard deviation of shadowing factor

xi_LOS = std_xi_LOS*randn(n_T, 1); % random shadowing 
xi_NLOS = std_xi_NLOS*randn(n_T, 1); % random shadowing
    

%  took from chao tang's function gen_IRS2BS
PL_NLOS = 35.3*log10(D3d)+22.4+21.3*log10(fc)-0.3*(h_Rx-1.5); %UMi-NLOS scenario: PL = 32.4 + 20log10(fc)+31.9log10(d_3d)
% PL_LOS = 32.4 + 21*log10(D3d) + 20*log10(fc);
% PL_NLOS = 32.4+20*log10(fc)+31.9*log10(D3d); %optional
d_BP = 4*h_Tx*h_Rx*(fc*1e9/c);
if dist_T2R < d_BP
    PL_LOS = 32.4 + 21*log10(D3d) + 20*log10(fc);
else
    PL_LOS = 32.4 + 40*log10(D3d) + 20*log10(fc) - 9.5*log10( d_BP^2 + (h_Tx-h_Rx)^2); 
end

PL_NLOS = max(PL_NLOS,PL_LOS);

% antenna_gain = 0; %(dB)
betta_LOS = ...
    ones(n_T,1)*...
    10^(antenna_gain/10)*...     % Tx antenna gain 
    10^(-PL_LOS/10).* ...        % path loss
    10.^(xi_LOS/10); 

betta_NLOS = ...
    ones(n_T,1)*...
    10^(antenna_gain/10)*...     % Tx antenna gain 
    10^(-PL_NLOS/10).* ...       % path loss
    10.^(xi_NLOS/10);  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% H
d_H_t       =   func_sig_vec(AoD_mean_H,n_T,deltan_t);
a_H_r       =   func_sig_vec(AoA_mean_H,n_R,deltan_r);
H_LOS       =   a_H_r * d_H_t.';
H_LOS       =   H_LOS .* exp(j*2*pi*ones(n_R,n_T) );%H_LOS .* ( exp(j*2*pi*(rand-0.5))*ones(n_R,n_T) ); % Incoporate random phase
H_LOS       =   H_LOS / norm(H_LOS,'fro');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% NLOS component: using 3GPP model
H_NLOS      =   func_scm_3gpp_static(AoD_mean_H,AoA_mean_H,AoD_AS_H,AoA_AS_H,n_T,n_R,deltan_t,deltan_r,M);
H_NLOS      =   H_NLOS / norm(H_NLOS,'fro');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% combing micro fading and macro fading

H_LOS = H_LOS * diag(sqrt(betta_LOS));
H_NLOS = H_NLOS * diag(sqrt(betta_NLOS));

%%
%%%%%%%%%%%% Combine LOS and NLOS
H_channel   =   sqrt(kappa_H/(kappa_H+1))*H_LOS + sqrt(1/(kappa_H+1))*H_NLOS;
% H_channel   =   H_channel / norm(H_channel,'fro');
%%%%%%%%%%%%