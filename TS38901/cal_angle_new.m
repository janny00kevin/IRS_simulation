function [AoD, ZoD, AoA,ZoA,d_hor] = cal_angle(loc_tx,loc_rx)

%% function Description
%{
This is a function to obtain AOA/AOD between Tx and Rx by their coordinates
INPUT:
    loc_tx, loc_rx are [x,y,z]' , (3x1)
    loc_tx : the coordinates of Transmiter
    loc_rx : the coordinates of Receiver

OUTPUT:
 AoD,AoA in [0,360] 

%}

%%


% direct distance between Tx and Rx
d_dir = norm(loc_tx - loc_rx);
% horizontal distance between Tx and Rx
d_hor = norm(loc_tx(1:2,1) - loc_rx(1:2,1));
% vertical distance between Tx and Rx
d_ver = norm(loc_tx(3,1) - loc_rx(3,1));
% 
% cos = d_hor/d_dir;
% 
% AoD = acosd(cos);
% AoA = 90 - AoD;
[~,AoD,ZoD] = xyz2pol(loc_rx-loc_tx);
[~,AoA,ZoA] = xyz2pol(loc_tx-loc_rx);

%% 
end