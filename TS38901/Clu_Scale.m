function C_Phi_NLOS = Clu_Scale(M)
% M : number of clusters
% table 7.5-2
    switch(M)
        case 4
            C_Phi_NLOS=0.779;
        case 5
            C_Phi_NLOS=0.860;
        case 8
            C_Phi_NLOS=1.018;
        case 10
            C_Phi_NLOS=1.09;
        case 11
            C_Phi_NLOS=1.123;
        case 12
            C_Phi_NLOS=1.146;
        case 14
            C_Phi_NLOS=1.19;
        case 15
            C_Phi_NLOS=1.211;
        case 16
            C_Phi_NLOS=1.226;
        case 19
            C_Phi_NLOS=1.273;
        case 20
            C_Phi_NLOS=1.289;
        case 25
            C_Phi_NLOS=1.358;
    end
end