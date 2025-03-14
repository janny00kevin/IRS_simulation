function C_Theta_NLOS = Clu_Scale_Z(M)
% M : number of clusters
% table 7.5-4
    switch(M)
        case 8
            C_Theta_NLOS=0.889;
        case 10
            C_Theta_NLOS=0.957;
        case 11
            C_Theta_NLOS=1.031;
        case 12
            C_Theta_NLOS=1.104;
        case 15
            C_Theta_NLOS=1.1088;
        case 19
            C_Theta_NLOS=1.184;
        case 20
            C_Theta_NLOS=1.178;
        case 25
            C_Theta_NLOS=1.282;
    end
end