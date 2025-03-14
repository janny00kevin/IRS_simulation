function H = chan_gen(loc_BSs,loc_UTs,ori_BSs,ori_UTs,nTxxy,nRxxy,f_arr,f_c,f_scenario,f_LOSProb,Tx_d_arr,Rx_d_arr,Tx_antenna_G,Rx_antenna_G,f_disable,room_size,r_clutter,h_clutter)
%
% Inputs:
arguments 
    loc_BSs (3,:) {mustBeNumeric,mustBeReal}
    % loc_BSs: Locations of all BSs in xyz-coor. 3 by nBS.
    loc_UTs (3,:) {mustBeNumeric,mustBeReal}
    % loc_UTs: Locations of all UTs in xyz-coor. 3 by nUE.
    ori_BSs (3,3,:) {mustBeNumeric,mustBeReal}
    % ori_BSs: orientations (in GCS) of all BSs. 3 by 3 by nBS . (normal direction,3 basis vector)
    ori_UTs (3,3,:) {mustBeNumeric,mustBeReal}
    % ori_UTs:  orientations (in GCS) of all UTs. 3 by 3 by nUT. (normal direction,3 basis vector)
    % (only used for UPA, ULA is assumed to be vertical to the ground. xy plane correspond to the UPA and z axis is the normal direction)
    nTxxy (1,2) {mustBeInteger,mustBePositive,mustBeNumeric}
    nRxxy (1,2) {mustBeInteger,mustBePositive,mustBeNumeric}
    % nTxxy,nRxxy: Tx/Rx antenna number on each side [x,y]
    f_arr (1,2) {mustBeNumeric,mustBeInteger,mustBeNonnegative}
    % f_arr: [(Tx array config) (Rx array config)] 0: single antenna, 1:ULA, 2:UPA
    f_c (1,1) {mustBeNumeric,mustBeReal}
    % f_c: carrier frequency
    f_scenario {mustBeMember(f_scenario,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
    % f_scenario: which scenario to use
    f_LOSProb {mustBeMember(f_LOSProb,["LOS","NLOS","TR"])}
    % f_LOSProb: 'LOS' for all links LOS, 'NLOS' for all links NLOS, 'TR' for
    % random assignment according to TS 38.901
    Tx_d_arr (1,2) {mustBeNumeric,mustBeReal}
    Rx_d_arr  (1,2) {mustBeNumeric,mustBeReal}
    % Tx_d_arr,RX_d_arr: [d_x,d_y] for UPA and d_x for ULA, in the unit of
    % signal wavelength (must be larger than 0.5)
    Tx_antenna_G  (1,1) {mustBeNumeric,mustBeReal}
    Rx_antenna_G (1,1) {mustBeNumeric,mustBeReal}
    % Tx_antenna_G,Rx_antenna_G : Tx/Rx antenna gain in dB
    f_disable (:,:) {mustBeNumericOrLogical,mustBeReal}
    % f_disable: nBS-by-nUE, binary, 1 indicate dead link (= 0)
    room_size (1,3) {mustBeNumeric,mustBeReal,mustBePositive} = [1 1 1]
    % only for InF scenario: hall size in meters [length width height]
    r_clutter (1,1) {mustBeNumeric,mustBeReal,mustBePositive} = 1
    % only for InF scenario: Clutter Density
    h_clutter (1,1) {mustBeNumeric,mustBeReal,mustBePositive} = 5
    % only for InF scenario: Effective Clutter Height







% b_UT_outdoor: nUT by 1. outdoor or indoor UE. (For UMi and UMa, 80% users
% should be indoor) //for now assume all UE outdoors

% H: channel coefficient
end

%% Constants
Ray_OS = [0.0447,-0.0447,0.1413,-0.1413,0.2492,-0.2492,0.3715,-0.3715,0.5129,-0.5129,...
    0.6797,-0.6797,0.8844,-0.8844,1.1481,-1.1481,1.5195,-1.5195,2.1551,-2.1551]; %table 7.5-3

Sub_Clu_1 = [1:8,19,20]; %table 7.5-5
Sub_Clu_2 = [9:12,17,18];
Sub_Clu_3 = [13:16];


%% General Parameters
nBS = size(loc_BSs,2); %num. of BS
nUT = size(loc_UTs,2); %num. of UE
H = cell(nBS,nUT);

nTx = nTxxy(1)*nTxxy(2);
nRx = nRxxy(1)*nRxxy(2);


% LOS_AOA,LOS_AOD, LOS_ZOA, LOS_ZOD
LOS_AOA = zeros(nBS,nUT);
LOS_AOD = zeros(nBS,nUT);
LOS_ZOA = zeros(nBS,nUT);
LOS_ZOD = zeros(nBS,nUT);
D_hor = zeros(nBS,nUT); %horizontal Distance
D_ver = zeros(nBS,nUT); %vertical Distance
for idxBS = 1:nBS
    for idxUT = 1:nUT
        [LOS_AOD(idxBS,idxUT),LOS_ZOD(idxBS,idxUT),LOS_AOA(idxBS,idxUT),...
            LOS_ZOA(idxBS,idxUT),D_hor(idxBS,idxUT),D_ver(idxBS,idxUT)] = cal_angle(loc_BSs(:,idxBS),loc_UTs(:,idxUT));
    end
end


%% 2. Large Sclae parameters

% Assign LOS/NLOS state of each BS-UT pair
prop_cond_LOS = zeros(nBS,nUT); % LOS or NLOS for each link, '1' for LOS and '0' for NLOS
switch lower(f_LOSProb)
    case 'los'
        prop_cond_LOS = ones(nBS,nUT);
    case 'nlos'
        prop_cond_LOS = zeros(nBS,nUT );
    case 'tr'
        rand_num = rand(nBS,nUT);
        prob_link = zeros(nBS,nUT);
        for idxBS = 1:nBS
            for idxUT = 1:nUT
                prob_link(idxBS,idxUT) = LOS_prob(f_scenario,D_hor(idxBS,idxUT),loc_UTs(3,idxUT),loc_BSs(3,idxBS),r_clutter,h_clutter);
                % Only consider all indoor/outdoor UEs, thus d_2d_out =
                % d_2d for uma/umi/rma. Will need other calculation if
                % indoor UTs are considered in this scenario.
            end
        end
        prop_cond_LOS = (rand_num>=prob_link);
end

% indoor/outdoor state of each BS-UT pair
% //for now assume all UT outdoors
% cond_UT = ones(nUT,1);

% Pathloss calculation
PL_LOS = zeros(nBS,nUT);
PL_NLOS = zeros(nBS,nUT);
sig_SF_NLOS = zeros(nBS,nUT);
sig_SF_LOS = zeros(nBS,nUT);
% SF_NLOS = zeros(nBS,nUT);
% SF_LOS = zeros(nBS,nUT);

for idxBS = 1:nBS
    for idxUT = 1:nUT
        [PL_NLOS(idxBS,idxUT),PL_LOS(idxBS,idxUT),sig_SF_NLOS(idxBS,idxUT),sig_SF_LOS(idxBS,idxUT)] = ...
            cal_PL(loc_BSs(:,idxBS),loc_UTs(:,idxUT),f_scenario,f_c);
    end
end
if 0
    disp(['PL_LOS=' num2str(mean(PL_LOS(:))) ', var=' num2str(var(PL_LOS(:)))]);
end

PL_NLOS = PL_NLOS + normrnd(0,sig_SF_NLOS,nBS,nUT) - (Tx_antenna_G+Rx_antenna_G);
PL_LOS = PL_LOS + normrnd(0,sig_SF_LOS,nBS,nUT)- (Tx_antenna_G+Rx_antenna_G);
PL_NLOS_Lin = 10.^(-PL_NLOS/10);
PL_LOS_Lin = 10.^(-PL_LOS/10);



% Generate LSP (ASA, ASD, ZSA, ZSD)
f_scenario = lower(f_scenario);
f_c_eff = f_c;
f_c_eff_ZSD = min(6,f_c);
if (strcmp(f_scenario,'umi') && f_c <2)
    f_c_eff = 2;
end
if (strcmp(f_scenario,'uma')&& f_c <6)
    f_c_eff = 6;
end



[ASD_LoS_mean,ASD_LoS_std,ASD_NLoS_mean,ASD_NLoS_std,~,~] = ASD_para_gen(f_c_eff,f_scenario);
[ASA_LoS_mean,ASA_LoS_std,ASA_NLoS_mean,ASA_NLoS_std,~,~] = ASA_para_gen(f_c_eff,f_scenario);
[ZSA_LoS_mean,ZSA_LoS_std,ZSA_NLoS_mean,ZSA_NLoS_std,~,~] = ZSA_para_gen(f_c_eff,f_scenario);
[Kappa_mean,Kappa_std] = Kappa_para_gen(f_scenario);
[nClu_LoS,nRay_LoS,nClu_NLoS,nRay_NLoS] = N_Clu_para_gen(f_scenario);
[C_ASD_LoS,C_ASA_LoS,C_ZSA_LoS,C_ASD_NLoS,C_ASA_NLoS,C_ZSA_NLoS] = C_A_para_gen(f_scenario);
[r_SF_var_LoS,r_SF_var_NLoS] = DS_scale_para_gen(f_scenario);
[Clu_SF_var_LoS,Clu_SF_var_NLoS] = Clu_SF_para_gen(f_scenario);
[DS_LoS_mean,DS_LoS_std,DS_NLoS_mean,DS_NLoS_std] = DS_para_gen(f_c_eff,f_scenario,room_size);

% in log scale
DS_LoS_all = normrnd(DS_LoS_mean,DS_LoS_std,nBS,nUT);
DS_NLoS_all = normrnd(DS_NLoS_mean,DS_NLoS_std,nBS,nUT);
ASD_LoS_all = normrnd(ASD_LoS_mean,ASD_LoS_std,nBS,nUT);
ASD_NLoS_all = normrnd(ASD_NLoS_mean,ASD_NLoS_std,nBS,nUT);
ASA_LoS_all = normrnd(ASA_LoS_mean,ASA_LoS_std,nBS,nUT);
ASA_NLoS_all = normrnd(ASA_NLoS_mean,ASA_NLoS_std,nBS,nUT);
ZSA_LoS_all = normrnd(ZSA_LoS_mean,ZSA_LoS_std,nBS,nUT);
ZSA_NLoS_all = normrnd(ZSA_NLoS_mean,ZSA_NLoS_std,nBS,nUT);
% in dB
Kappa = normrnd(Kappa_mean,Kappa_std,nBS,nUT);

% in linear scale
DS_LoS_all = 10.^DS_LoS_all;
DS_NLoS_all = 10.^DS_NLoS_all;
ASD_LoS_all = 10.^ASD_LoS_all;
ASD_NLoS_all = 10.^ASD_NLoS_all;
ASA_LoS_all = 10.^ASA_LoS_all;
ASA_NLoS_all = 10.^ASA_NLoS_all;
ZSA_LoS_all = 10.^ZSA_LoS_all;
ZSA_NLoS_all = 10.^ZSA_NLoS_all;

ASD_LoS_all = min(ASD_LoS_all,104);
ASD_NLoS_all = min(ASD_NLoS_all,104);
ASA_LoS_all =  min(ASA_LoS_all,104);
ASA_NLoS_all =  min(ASA_NLoS_all,104);
ZSA_LoS_all = min(52,ZSA_LoS_all);
ZSA_NLoS_all = min(52,ZSA_NLoS_all);

Kappa_lin = 10.^(Kappa/10);




for idxBS = 1:nBS
    for idxUT = 1:nUT
        if(f_disable(idxBS,idxUT)==1)
            H{idxBS,idxUT} = zeros(nRx,nTx);
        else
        [ZSD_LoS_mean_temp,ZSD_LoS_std_temp, ZSD_LoS_OS_temp,ZSD_NLoS_mean_temp,ZSD_NLoS_std_temp, ZSD_NLoS_OS_temp] = ...
            ZSD_para_gen(f_c_eff_ZSD,f_scenario,loc_BSs(:,idxBS),loc_UTs(:,idxUT));

        
        if(prop_cond_LOS(idxBS,idxUT)) % if link LOS: LOS conponent
            nRay = nRay_LoS;
            C_ASA = C_ASA_LoS;
            C_ASD = C_ASD_LoS;
            C_ZSA = C_ZSA_LoS;
            ZSD_mean =ZSD_LoS_mean_temp;
            
            % generate cluster delays:(only what needed for cluster power generation
            X = rand([nClu_LoS,1]);
            tau_LOS_ = -r_SF_var_LoS*DS_LoS_all(idxBS,idxUT).*log(X);
            tau_LOS = sort(tau_LOS_-min(tau_LOS_));
            
            % generate cluster power
            Z = normrnd(0,Clu_SF_var_LoS,[nClu_LoS,1]);
            P_LOS_ = exp(-tau_LOS.*(r_SF_var_LoS-1)/(r_SF_var_LoS*DS_LoS_all(idxBS,idxUT))).*10.^(-Z/10);
            K_R = Kappa_lin(idxBS,idxUT);
            P_LOS_1 =  K_R/(K_R+1) ;
            P_CLU = (1/K_R+1)*P_LOS_/sum(P_LOS_);
            P_CLU(1) = P_CLU(1) + P_LOS_1;
            P_CLU = P_CLU(P_CLU/max(P_CLU) >= 10^(-2.5));
            nClu_eff = length(P_CLU);
            
            
            % generate AOA/AOD
            K = Kappa(idxBS,idxUT);
            C_Phi_LOS = Clu_Scale(nClu_LoS)*(1.1035-0.028*K-0.002*K^2+0.0001*K^3);
            Phi_ASA_ = 2*ASA_LoS_all(idxBS,idxUT)/1.4*sqrt(-log(P_CLU/max(P_CLU)))/C_Phi_LOS;
            X = zeros(nClu_eff,1);
            X(rand(nClu_eff,1)<0.5) = -1;
            Y = normrnd(0,ASA_LoS_all(idxBS,idxUT)/7,nClu_eff,1);
            %             Phi_ASA_NLOS = X.*Phi_ASA_+ Y+ LOS_AOA;
            Phi_AoA = (X.*Phi_ASA_+ Y) - (X(1)*Phi_ASA_(1)+Y(1)-LOS_AOA(idxBS,idxUT));
            
            
            Phi_ASD_ = 2*ASD_LoS_all(idxBS,idxUT)/1.4*sqrt(-log(P_CLU/max(P_CLU)))/C_Phi_LOS;
            X = ones(nClu_eff,1);
            X(rand(nClu_eff,1)<0.5) = -1;
            Y = normrnd(0,ASD_LoS_all(idxBS,idxUT)/7,nClu_eff,1);
            Phi_AoD = (X.*Phi_ASD_+ Y) - (X(1)*Phi_ASD_(1)+Y(1)-LOS_AOD(idxBS,idxUT));
            
            % generate ZOA
            C_Theta_LOS = Clu_Scale_Z(nClu_LoS)*(1.3086-0.0339*K-0.0077*K^2+0.0002*K^3);
            Theta_ZOA_ = -ZSA_LoS_all(idxBS,idxUT)*log(P_CLU/max(P_CLU))/C_Theta_LOS;
            X = ones(nClu_eff,1);
            X(rand(nClu_eff,1)<0.5) = -1;
            Y = normrnd(0,ZSA_LoS_all(idxBS,idxUT)/7,nClu_eff,1);
            % if O2I.....(Not implemented, check the TR)
            Theta_ZOA = (X.*Theta_ZOA_+ Y) - (X(1)*Theta_ZOA_(1)+Y(1)-LOS_ZOA(idxBS,idxUT)); % rounding
            
            % generate ZOD
            ZSD_LoS = normrnd(ZSD_LoS_mean_temp,ZSD_LoS_std_temp);
            ZSD_LoS = 10.^ZSD_LoS;
            ZSD_LoS = min(52,ZSD_LoS);
            
            Theta_ZOD_ = -ZSD_LoS*log(P_CLU/max(P_CLU))/C_Theta_LOS;
            X = ones(nClu_eff,1);
            X(rand(nClu_eff,1)<0.5) = -1;
            Y = normrnd(0,ZSA_LoS_all(idxBS,idxUT)/7,nClu_eff,1);
            Theta_ZOD = (X.*Theta_ZOD_+ Y) - (X(1)*Theta_ZOD_(1)+Y(1)-LOS_ZOD(idxBS,idxUT));
              
        else %NLOS
            nRay = nRay_NLoS;
            C_ASA = C_ASA_NLoS;
            C_ASD = C_ASD_NLoS;
            C_ZSA = C_ZSA_NLoS;
            ZSD_mean =ZSD_NLoS_mean_temp;
            % generate cluster delays:(only what needed for cluster power generation
            X = rand([nClu_NLoS,1]);
            tau_LOS_ = -r_SF_var_NLoS*DS_NLoS_all(idxBS,idxUT).*log(X);
            tau_LOS = sort(tau_LOS_-min(tau_LOS_));
            
            % generate cluster power
            Z = normrnd(0,Clu_SF_var_NLoS,[nClu_NLoS,1]);
            P_LOS_ = exp(-tau_LOS.*(r_SF_var_NLoS-1)/(r_SF_var_NLoS*DS_NLoS_all(idxBS,idxUT))).*10.^(-Z/10);
            K_R = Kappa_lin(idxBS,idxUT);
%             P_LOS_1 =  K_R/(K_R+1) ;
            P_CLU = (1/K_R+1)*P_LOS_/sum(P_LOS_);
            P_CLU = P_CLU(P_CLU/max(P_CLU) >= 10^(-2.5));
            nClu_eff = length(P_CLU);
            
            % generate AOA/AOD
%             K = Kappa(idxBS,idxUT);
            C_Phi_NLOS = Clu_Scale(nClu_NLoS);
            Phi_ASA_ = 2*ASA_NLoS_all(idxBS,idxUT)/1.4*sqrt(-log(P_CLU/max(P_CLU)))/C_Phi_NLOS;
            X = zeros(nClu_eff,1);
            X(rand(nClu_eff,1)<0.5) = -1;
            Y = normrnd(0,ASA_NLoS_all(idxBS,idxUT)/7,nClu_eff,1);
            %             Phi_ASA_NLOS = X.*Phi_ASA_+ Y+ LOS_AOA;
            Phi_AoA = X.*Phi_ASA_+ Y + LOS_AOA(idxBS,idxUT);
            
            Phi_ASD_ = 2*ASD_NLoS_all(idxBS,idxUT)/1.4*sqrt(-log(P_CLU/max(P_CLU)))/C_Phi_NLOS;
            X = ones(nClu_eff,1);
            X(rand(nClu_eff,1)<0.5) = -1;
            Y = normrnd(0,ASD_NLoS_all(idxBS,idxUT)/7,nClu_eff,1);
            Phi_AoD = X.*Phi_ASD_+ Y + LOS_AOD(idxBS,idxUT);
            
            % generate ZOA
            C_Theta_NLOS = Clu_Scale_Z(nClu_NLoS);
            Theta_ZOA_ = -ZSA_NLoS_all(idxBS,idxUT)*log(P_CLU/max(P_CLU))/C_Theta_NLOS;
            X = ones(nClu_eff,1);
            X(rand(nClu_eff,1)<0.5) = -1;
            Y = normrnd(0,ZSA_NLoS_all(idxBS,idxUT)/7,nClu_eff,1);
            % if O2I.....(Not implemented, check the TR)
            Theta_ZOA = X.*Theta_ZOA_+ Y +LOS_ZOA(idxBS,idxUT); % rounding
           
            
            % generate ZOD
            ZSD_NLoS = normrnd(ZSD_NLoS_OS_temp,ZSD_NLoS_mean_temp);
            ZSD_NLoS = 10.^ZSD_NLoS;
            ZSD_NLoS = min(52,ZSD_NLoS);
            
            Theta_ZOD_ = -ZSD_NLoS*log(P_CLU/max(P_CLU))/C_Theta_NLOS;
            X = ones(nClu_eff,1);
            X(rand(nClu_eff,1)<0.5) = -1;
            Y = normrnd(0,ZSA_LoS_all(idxBS,idxUT)/7,nClu_eff,1);
            Theta_ZOD = X.*Theta_ZOD_+ Y+ LOS_ZOD(idxBS,idxUT) + ZSD_NLoS_std_temp;
            
            
        end
        
        
        H_NLoS = zeros(nRx,nTx);
        H_LoS = zeros(nRx,nTx);
        for idxClu = 1:nClu_eff
            % ray angles
            Phi_AOA_Ray = Phi_AoA(idxClu) + C_ASA*Ray_OS(1:nRay);
            Phi_AOD_Ray = Phi_AoD(idxClu) + C_ASD*Ray_OS(1:nRay);
            Theta_ZOA_Ray = Theta_ZOA(idxClu) + C_ZSA*Ray_OS(1:nRay);
            Theta_ZOD_Ray = Theta_ZOD(idxClu) + (0.375)*10^ZSD_mean.*Ray_OS(1:nRay);
            Theta_ZOA_Ray(Theta_ZOA_Ray>=180) = 360-Theta_ZOA_Ray(Theta_ZOA_Ray>=180);
            Theta_ZOD_Ray(Theta_ZOD_Ray>=180) = 360-Theta_ZOD_Ray(Theta_ZOD_Ray>=180);
            
            % random initial phase
            ini_phase = pi*randn(nRay,1);
            if (prop_cond_LOS(idxBS,idxUT) && idxClu==1 )
                ini_phase = ones(nRay,1);  %% should it be zros(.)?????
            end
            
            % random A/D mapping
            if (idxClu ==1 || idxClu ==2)                
                Phi_AOD_Ray(Sub_Clu_1) = Phi_AOD_Ray(Sub_Clu_1(randperm(10)));
                Phi_AOD_Ray(Sub_Clu_2) = Phi_AOD_Ray(Sub_Clu_2(randperm(6)));
                Phi_AOD_Ray(Sub_Clu_3) = Phi_AOD_Ray(Sub_Clu_3(randperm(4)));
                
                Theta_ZOD_Ray(Sub_Clu_1) = Theta_ZOD_Ray(Sub_Clu_1(randperm(10)));
                Theta_ZOD_Ray(Sub_Clu_2) = Theta_ZOD_Ray(Sub_Clu_2(randperm(6)));
                Theta_ZOD_Ray(Sub_Clu_3) = Theta_ZOD_Ray(Sub_Clu_3(randperm(4)));
                
                % channel coeff
                H_Clu = zeros(nRx,nTx);
                for idxRay = Sub_Clu_1
                    % the angel calculation here is very simplified with
                    % many assumptions!

                    % Tx
                    if(f_arr(1) == 0) % single antenna
                        a_Tx = 1;
                    elseif (f_arr(1) == 1)% ULA                       
                        Tx_Theta = 90-Theta_ZOD_Ray(idxRay);
                        a_Tx = str_ULA(Tx_Theta,Tx_d_arr(1),nTx);
                    else % UPA
                        A = ori_BSs(:,:,idxBS);
                        r_x = Ang2r(Theta_ZOD_Ray(idxRay),Phi_AOD_Ray(idxRay));
                        r_x_proj = A/(A'*A)*A'*r_x; %project onto LCS of the device 
                        [~,Tx_Phi,Tx_Theta] = xyz2pol(r_x_proj);
                         a_Tx = str_UPA(Tx_Phi,Tx_Theta,Tx_d_arr(1),Tx_d_arr(2),nTxxy(1),nTxxy(2));
                    end
                    
                    % Rx
                    if(f_arr(2) == 0) % single antenna
                        a_Rx = 1;
                    elseif (f_arr(2) == 1)% ULA   
                        Rx_Theta = abs(Theta_ZOA_Ray(idxRay)-90);
%                         if(Rx_Theta>90)
%                             Rx_Theta = Theta_ZOA_Ray(idxRay)-90;
%                         else
%                             Rx_Theta = 90-Theta_ZOA_Ray(idxRay);
%                         end                        
                        a_Rx = str_ULA(Rx_Theta,Rx_d_arr(1),nRx);
                    else % UPA
                        A = ori_UTs(:,:,idxUT);
                        r_x = Ang2r(Theta_ZOA_Ray(idxRay),Phi_AOA_Ray(idxRay));
                        r_x_proj = A/(A'*A)*A'*r_x;
                        [~,Rx_Phi,Rx_Theta] = xyz2pol(r_x_proj);
                         a_Rx = str_UPA(Rx_Phi,Rx_Theta,Rx_d_arr(1),Rx_d_arr(2),nRxxy(1),nRxxy(2));
                    end
                        H_Clu = H_Clu + exp(1j*ini_phase(idxRay))*a_Rx*a_Tx.';                    
                end
                if(prop_cond_LOS(idxBS,idxUT) && idxClu==1)
                    H_LoS = H_LoS + sqrt(P_CLU(idxClu)*10/nRay/nRay)*H_Clu;
                else
                     H_NLoS = H_NLoS + sqrt(P_CLU(idxClu)*10/nRay/nRay)*H_Clu;
                end
                
                H_Clu = zeros(nRx,nTx);
                 for idxRay = Sub_Clu_2
                    % the angel calculation here is very simplified with
                    % many assumption!
                    %Tx
                    if(f_arr(1) == 0) % single antenna
                        a_Tx = 1;
                    elseif (f_arr(1) == 1)% ULA
                        if(Tx_Theta>90)
                            Tx_Theta = Theta_ZOD_Ray(idxRay)-90;
                        else
                            Tx_Theta = 90-Theta_ZOD_Ray(idxRay);
                        end
                        
                        a_Tx = str_ULA(Tx_Theta,Tx_d_arr(1),nTx);
                    else % UPA
                        A = ori_BSs(:,:,idxBS);
                        r_x = Ang2r(Theta_ZOD_Ray(idxRay),Phi_AOD_Ray(idxRay));
                        r_x_proj = A/(A'*A)*A'*r_x;
                        [~,Tx_Phi,Tx_Theta] = xyz2pol(r_x_proj);
                         a_Tx = str_UPA(Tx_Phi,Tx_Theta,Tx_d_arr(1),Tx_d_arr(2),nTxxy(1),nTxxy(2));
                    end
                    
                    % 
                    %Rx
                    if(f_arr(2) == 0) % single antenna
                        a_Rx = 1;
                    elseif (f_arr(2) == 1)% ULA            
                        Rx_Theta = abs(Theta_ZOA_Ray(idxRay)-90);
%                         if(Rx_Theta>90)
%                             Rx_Theta = Theta_ZOA_Ray(idxRay)-90;
%                         else
%                             Rx_Theta = 90-Theta_ZOA_Ray(idxRay);
%                         end
                        
                        a_Rx = str_ULA(Rx_Theta,Rx_d_arr(1),nRx);
                    else % UPA
                        A = ori_UTs(:,:,idxUT);
                        r_x = Ang2r(Theta_ZOA_Ray(idxRay),Phi_AOA_Ray(idxRay));
                        r_x_proj = A/(A'*A)*A'*r_x;
                        [~,Rx_Phi,Rx_Theta] = xyz2pol(r_x_proj);
                         a_Rx = str_UPA(Rx_Phi,Rx_Theta,Rx_d_arr(1),Rx_d_arr(2),nRxxy(1),nRxxy(2));
                    end
                        H_Clu = H_Clu + exp(1j*ini_phase(idxRay))*a_Rx*a_Tx.';                    
                 end
                if(prop_cond_LOS(idxBS,idxUT) && idxClu==1)
                    H_LoS = H_LoS + sqrt(P_CLU(idxClu)*6/nRay/nRay)*H_Clu;
                    else
                     H_NLoS = H_NLoS + sqrt(P_CLU(idxClu)*6/nRay/nRay)*H_Clu;
                end
                
                H_Clu = zeros(nRx,nTx);
                 for idxRay = Sub_Clu_3
                    % the angel calculation here is very simplified with
                    % many assumption!
                    %Tx
                    if(f_arr(1) == 0) % single antenna
                        a_Tx = 1;
                    elseif (f_arr(1) == 1)% ULA                       
                        if(Tx_Theta>90)
                            Tx_Theta = Theta_ZOD_Ray(idxRay)-90;
                        else
                            Tx_Theta = 90-Theta_ZOD_Ray(idxRay);
                        end
                        a_Tx = str_ULA(Tx_Theta,Tx_d_arr(1),nTx);
                    else % UPA
                        A = ori_BSs(:,:,idxBS);
                        r_x = Ang2r(Theta_ZOD_Ray(idxRay),Phi_AOD_Ray(idxRay));
                        r_x_proj = A/(A'*A)*A'*r_x;
                        [~,Tx_Phi,Tx_Theta] = xyz2pol(r_x_proj);
                         a_Tx = str_UPA(Tx_Phi,Tx_Theta,Tx_d_arr(1),Tx_d_arr(2),nTxxy(1),nTxxy(2));
                    end
                    
                    % 
                    %Rx
                    if(f_arr(2) == 0) % single antenna
                        a_Rx = 1;
                    elseif (f_arr(2) == 1)% ULA                       
                        if(Rx_Theta>90)
                            Rx_Theta = Theta_ZOA_Ray(idxRay)-90;
                        else
                            Rx_Theta = 90-Theta_ZOA_Ray(idxRay);
                        end
                        
                        a_Rx = str_ULA(Rx_Theta,Rx_d_arr(1),nRx);
                    else % UPA
                        A = ori_UTs(:,:,idxUT);
                        r_x = Ang2r(Theta_ZOA_Ray(idxRay),Phi_AOA_Ray(idxRay));
                        r_x_proj = A/(A'*A)*A'*r_x;
                        [~,Rx_Phi,Rx_Theta] = xyz2pol(r_x_proj);
                         a_Rx = str_UPA(Rx_Phi,Rx_Theta,Rx_d_arr(1),Rx_d_arr(2),nRxxy(1),nRxxy(2));
                    end
                        H_Clu = H_Clu + exp(1j*ini_phase(idxRay))*a_Rx*a_Tx.';                    
                 end
                if(prop_cond_LOS(idxBS,idxUT) && idxClu==1)
                    H_LoS = H_LoS + sqrt(P_CLU(idxClu)*4/nRay/nRay)*H_Clu;
                    else
                     H_NLoS = H_NLoS + sqrt(P_CLU(idxClu)*4/nRay/nRay)*H_Clu;
                end
            else
                %clust 3~ or more
                % A-D pairing
                Phi_AOD_Ray = Phi_AOD_Ray(randperm(nRay));
                Theta_ZOD_Ray = Theta_ZOD_Ray(randperm(nRay)); 
                 H_Clu = zeros(nRx,nTx);
                for idxRay = 1:nRay
                    % the angel calculation here is very simplified with
                    % many assumption!
                    %Tx
                    if(f_arr(1) == 0) % single antenna
                        a_Tx = 1;
                    elseif (f_arr(1) == 1)% ULA                       
                        if(Tx_Theta>90)
                            Tx_Theta = Theta_ZOD_Ray(idxRay)-90;
                        else
                            Tx_Theta = 90-Theta_ZOD_Ray(idxRay);
                        end
                        
                        a_Tx = str_ULA(Tx_Theta,Tx_d_arr(1),nTx);
                    else % UPA
                        A = ori_BSs(:,:,idxBS);
                        r_x = Ang2r(Theta_ZOD_Ray(idxRay),Phi_AOD_Ray(idxRay));
                        r_x_proj = A/(A'*A)*A'*r_x;
                        [~,Tx_Phi,Tx_Theta] = xyz2pol(r_x_proj);
                         a_Tx = str_UPA(Tx_Phi,Tx_Theta,Tx_d_arr(1),Tx_d_arr(2),nTxxy(1),nTxxy(2));
                    end
                    
                    % 
                    %Rx
                    if(f_arr(2) == 0) % single antenna
                        a_Rx = 1;
                    elseif (f_arr(2) == 1)% ULA                       
                        if(Rx_Theta>90)
                            Rx_Theta = Theta_ZOA_Ray(idxRay)-90;
                        else
                            Rx_Theta = 90-Theta_ZOA_Ray(idxRay);
                        end
                        
                        a_Rx = str_ULA(Rx_Theta,Rx_d_arr(1),nRx);
                    else % UPA
                        A = ori_UTs(:,:,idxUT);
                        r_x = Ang2r(Theta_ZOA_Ray(idxRay),Phi_AOA_Ray(idxRay));
                        r_x_proj = A/(A'*A)*A'*r_x;
                        [~,Rx_Phi,Rx_Theta] = xyz2pol(r_x_proj);
                         a_Rx = str_UPA(Rx_Phi,Rx_Theta,Rx_d_arr(1),Rx_d_arr(2),nRxxy(1),nRxxy(2));
                    end
                        H_Clu = H_Clu + exp(1j*ini_phase(idxRay))*a_Rx*a_Tx.';                    
                end
                if(prop_cond_LOS(idxBS,idxUT) && idxClu==1)
                    H_LoS = H_LoS + sqrt(P_CLU(idxClu)/nRay)*H_Clu;
                else
                     H_NLoS = H_NLoS + sqrt(P_CLU(idxClu)/nRay)*H_Clu;
                end
                
            end
          
        end
        H{idxBS,idxUT} = sqrt(PL_LOS_Lin(idxBS,idxUT))*sqrt(Kappa_lin(idxBS,idxUT)/(Kappa_lin(idxBS,idxUT)+1))*H_LoS+...
            sqrt(PL_NLOS_Lin(idxBS,idxUT))*sqrt(1/(Kappa_lin(idxBS,idxUT)+1))*H_NLoS;
        end
    end
end
end


