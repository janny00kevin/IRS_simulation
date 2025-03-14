function H = GENERATE_CHANNEL(CH_MODEL, SCENARIO, LOS, f_c)
    switch lower(CH_MODEL)
        case 'rayleigh-fading'
            AoD_mean_H = 45/180*pi;
            AoA_mean_H = 45/180*pi;
            AoD_AS_H = 10/180*pi;
            AoA_AS_H = pi;
            nTx = 36;
            nRx = 4;
            kappa_H = 0;
            dist_T2R = 20;
            h_Tx = 10;
            h_Rx = 2;
            antenna_gain = 8;
            c = 3e8;
%             fc = 10;
            lambda_c = c/(fc*1e9);
            delta_t = lambda_c/2;
            delta_r = lambda_c/2;
            M = 20;     % number of path
                
            % there is a comment out in line 67-68 of func_3gpp_scm_w_LOS.m since
            % considering only the micro fading part
            H = func_3gpp_scm_w_LOS(AoD_mean_H, AoA_mean_H, AoD_AS_H, AoA_AS_H, ...
                nTx, nRx, kappa_H, dist_T2R, h_Tx, h_Rx, antenna_gain, delta_t, ...
                delta_r, M);

        case 'ct'
            % pick scenario
%             scenario = 'inf-dl';
            switch lower(SCENARIO)
                case 'rma'
                    %% General setting for channel generation
                    loc_BSs = [500; 1000; 35];
                    loc_UTs = [900; 450; 1.5];     % set distance between BSs and UT to be around 680 m in 2D [min 35 m in 2D]
                    ori_BSs = 1;
                    ori_UTs = [1 0 0;0 0 1;0 1 0];
                    nRxxy = [2 2];
                    nRx = 4;
                    f_arr = [1 2];
%                     f_c = 2.5;
                    f_scenario = scenario;
                    f_LOSProb = LOS;
                    Tx_d_arr = 0.5;
                    Rx_d_arr = [0.5 0.5];
                    Tx_antenna_G = 8;
                    Rx_antenna_G = 8;
                    f_disable = [0; 0];
                    room_size = [1732 1732 1732];  % ISD = 1732m according to table 7.2-3 for RMa TR38.901v16
                    r_clutter = 1;
                    h_clutter = 5;
                    %% Channel of dimension 4x36
                    ntx = 36;
                    nTxxy = [ntx, 1]; % [36, 48, 100]

                case 'inf-dl'
                    %% General setting for channel generation
%                     nBS = 1;
%                     nIRS = 1;
                    nRxxy = [2 2];
                    ntx = 36;
                    nTxxy = [ntx, 1];
%                     d_BS_UT_L = 15;
%                     d_y = 50; 
%                     h_BS = 10;
%                     h_IRS = 10;
%                     [~,loc_IRSs,ori_IRSs] = loc_gen_BS_line(nBS,nIRS,d_BS_UT_L,d_y,h_BS,h_IRS);
%                     nUE = 1;
%                     d_BS_IRS_L = 15;
%                     d_IRS_UE_L = 90;
%                     d_UE_L = 15;
%                     h_UE = 2;
%                     [loc_UE] = loc_gen_UE_line(nUE,d_BS_IRS_L,d_IRS_UE_L,d_UE_L,d_y,h_UE);
                    
%                     f_disable = zeros(nIRS,nUE);
%                     for idxUE = 1:nUE
%                         loc_UE_curr = loc_UE(:,idxUE);
%                         for idxIRS = 1:nIRS
%                             loc_IRS_curr = loc_IRSs(:,idxIRS);
%                             if((loc_UE_curr-loc_IRS_curr)'*ori_IRSs(:,3,idxIRS)<0)
%                                 f_disable(idxIRS,idxUE) = 1;
%                             end
%                         end
%                     end

                                
                    ori_BSs = 1;
                    f_arr = [1 1];
%                     f_c = 2.5;
                    f_scenario = SCENARIO;
                    f_LOSProb = LOS;
                    Tx_d_arr = 0.5;
                    Rx_d_arr = [0.5 0.5];
                    Tx_antenna_G = 8;
                    Rx_antenna_G = 5;
%                     f_disable = [0; 0];
                    room_size = [120 50 15]; % [300 150 10]
                    r_clutter = 0.4;
                    h_clutter = 10;

%                     loc_BSs = loc_IRSs;
%                     loc_UTs = loc_UE;
%                     ori_UTs = ori_IRSs;

                    % AFTER SOME REPEATED CALCULATION
                    loc_BSs = [15; 0; 10];
                    loc_UTs = [110; 21; 2];
                    ori_UTs = [1 0 0; 0 0 1; 0 1 0];
                    f_disable = 0;

                case 'uma'
                    %% General setting for channel generation
                    loc_BSs = [500; 1000; 25];
                    loc_UTs = [900; 450; 1.5];     % set distance between BSs and UT to be around 680 m in 2D [min 35 m in 2D]
                    ori_BSs = 1;
                    ori_UTs = [1 0 0;0 0 1;0 1 0];
                    nRxxy = [2 2];
                    nRx = 4;
                    f_arr = [1 1];
%                     f_c = 2.5;
                    f_scenario = SCENARIO;
                    f_LOSProb = LOS;
                    Tx_d_arr = 0.5;
                    Rx_d_arr = [0.5 0.5];
                    Tx_antenna_G = 8;
                    Rx_antenna_G = 8;
                    f_disable = [0; 0];
                    room_size = [500 500 500];  % ISD = 500m according to table 7.2-1 for RMa TR38.901v16
                    r_clutter = 1;
                    h_clutter = 5;
                    %% Channel of dimension 4x36
                    ntx = 36;
                    nTxxy = [ntx, 1]; % [36, 48, 100]
            end
        
            % get channel realization
            H = chan_gen(loc_BSs,loc_UTs,ori_BSs,ori_UTs,nTxxy,nRxxy,f_arr,f_c,f_scenario,f_LOSProb,Tx_d_arr,Rx_d_arr,Tx_antenna_G,Rx_antenna_G,f_disable,room_size,r_clutter,h_clutter);
            H = H{1};
            while abs(H(1)) > 1e-4 || abs(H(1)) < 1e-6
                H = chan_gen(loc_BSs,loc_UTs,ori_BSs,ori_UTs,nTxxy,nRxxy,f_arr,f_c,f_scenario,f_LOSProb,Tx_d_arr,Rx_d_arr,Tx_antenna_G,Rx_antenna_G,f_disable,room_size,r_clutter,h_clutter);
                H = H{1};
            end
    end
end








