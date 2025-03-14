function H = GENERATE_CHANNEL_IU(CH_MODEL, SCENARIO, LOS, f_c)
    % General setting for channel generation
    loc_BSs = [51; 0; 0];
    loc_UTs = [48.7639320225; -2; 0];     % set distance between BSs and UT to be around 680 m in 2D [min 35 m in 2D]
    ori_BSs = [0 0 -1;0 1 0;1 0 0];
    ori_UTs = [0 -1 0;0 0 1;-1 0 0];
    nTxxy = [4 2];
    nRxxy = [2 2];
    f_arr = [2 2];
    f_scenario = SCENARIO;
    f_LOSProb = LOS;
    Tx_d_arr = [0.5 0.5];
    Rx_d_arr = [0.5 0.5];
    Tx_antenna_G = 8;
    Rx_antenna_G = 8;
    f_disable = [0; 0];
    room_size = [500 500 500];  % ISD = 500m according to table 7.2-1 for RMa TR38.901v16
    r_clutter = 1;
    h_clutter = 5;
    % Channel of dimension 4x36 % [36, 48, 100]
    
    % get channel realization
    H = chan_gen(loc_BSs,loc_UTs,ori_BSs,ori_UTs,nTxxy,nRxxy,f_arr,f_c,f_scenario,f_LOSProb,Tx_d_arr,Rx_d_arr,Tx_antenna_G,Rx_antenna_G,f_disable,room_size,r_clutter,h_clutter);
    H = H{1};
    while abs(H(1)) > 1e-4 || abs(H(1)) < 1e-6
        H = chan_gen(loc_BSs,loc_UTs,ori_BSs,ori_UTs,nTxxy,nRxxy,f_arr,f_c,f_scenario,f_LOSProb,Tx_d_arr,Rx_d_arr,Tx_antenna_G,Rx_antenna_G,f_disable,room_size,r_clutter,h_clutter);
        H = H{1};
    end
%     disp(H)