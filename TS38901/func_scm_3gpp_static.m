function H_coeff = func_scm_3gpp_static(AoD_mean,AoA_mean,AoD_AS,AoA_AS,Nt,Nr,delta_t,delta_r,M)
%  AS: Angular Spread
% Nt/Nr: no. of antenna
% delta: inter antenna distance
% M: number of pathes

theta_Tx    =   AoD_mean;
theta_Rx    =   AoA_mean;
sigma_AoD   =   AoD_AS;
sigma_AoA   =   AoA_AS ;

% M = 120; N=1;

theta_AoD   =   theta_Tx * ones(M,1) + sigma_AoD * ( randn(M,1) )   ; %     
%theta_AoD   =   theta_Tx * ones(M,1) + sigma_AoD; %* ( randn(M,1) )   ;
theta_AoD   =   sort(theta_AoD);

theta_AoA   =   theta_Rx * ones(M,1) + sigma_AoA * ( randn(M,1) )  ; % -0.5*ones(M,1)
%theta_AoA   =   theta_Rx * ones(M,1) + sigma_AoA; %* ( randn(M,1) )  ;
theta_AoA   =   sort(theta_AoA);
H_temp      =   zeros(Nr,Nt,M);

for m = 1:M
    sig_vec_dep(:,m)    =   func_sig_vec(theta_AoD(m),Nt,delta_t); % Departure signature vector:.* exp(j*2*pi*rand*ones(Nt,1))
    sig_vec_arr(:,m)    =   func_sig_vec(theta_AoA(m),Nr,delta_r); % Arrival signature vector
    
    H_temp(:,:,m)       =   sig_vec_arr(:,m) * sig_vec_dep(:,m).';
    H_temp(:,:,m)       =   H_temp(:,:,m) .* ( exp(1j*2*pi*(rand-0.5))*ones(Nr,Nt) ); % Random Phase
    H_temp(:,:,m)       =   H_temp(:,:,m) / sqrt(Nr*Nt);  
end
H_coeff     =   sum(H_temp,3); %/sqrt(M) ;
H_coeff     =   H_coeff/norm(H_coeff,'fro');
end




