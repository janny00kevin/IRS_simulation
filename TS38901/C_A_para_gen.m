function [C_ASD_LoS,C_ASA_LoS,C_ZSA_LoS,C_ASD_NLoS,C_ASA_NLoS,C_ZSA_NLoS, C_ASD_O2I, C_ASA_O2I, C_ZSA_O2I] = C_A_para_gen(kase)
% generate Cluster ASD/ASA/ZSA (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase:{mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor


    switch lower(kase)
        case 'umi'
            C_ASD_LoS = 3;
            C_ASA_LoS = 17;
            C_ZSA_LoS = 7;
            C_ASD_NLoS = 10;
            C_ASA_NLoS = 22;
            C_ZSA_NLoS = 7;
            C_ASD_O2I = 5;
            C_ASA_O2I = 8;
            C_ZSA_O2I = 3;
        case 'uma'
            C_ASD_LoS = 5;
            C_ASA_LoS = 11;
            C_ZSA_LoS = 7;
            C_ASD_NLoS = 2;
            C_ASA_NLoS = 15;
            C_ZSA_NLoS = 7;
            C_ASD_O2I = 5;
            C_ASA_O2I = 8;
            C_ZSA_O2I = 3;
        case 'rma'
            C_ASD_LoS = 2;
            C_ASA_LoS = 3;
            C_ZSA_LoS = 3;
            C_ASD_NLoS = 2;
            C_ASA_NLoS = 3;
            C_ZSA_NLoS = 3;
            C_ASD_O2I = 2;
            C_ASA_O2I = 3;
            C_ZSA_O2I = 3;
        case 'inoff'
            C_ASD_LoS = 5;
            C_ASA_LoS = 8;
            C_ZSA_LoS = 9;
            C_ASD_NLoS = 5;
            C_ASA_NLoS = 11;
            C_ZSA_NLoS = 9;
            C_ASD_O2I = NaN;
            C_ASA_O2I = NaN;
            C_ZSA_O2I = NaN;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            C_ASD_LoS = 5;
            C_ASA_LoS = 8;
            C_ZSA_LoS = 9;
            C_ASD_NLoS = 5;
            C_ASA_NLoS = 8;
            C_ZSA_NLoS = 9;
            C_ASD_O2I = NaN;
            C_ASA_O2I = NaN;
            C_ZSA_O2I = NaN;
    end
end