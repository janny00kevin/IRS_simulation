function [ZSA_LoS_mean,ZSA_LoS_var,ZSA_NLoS_mean,ZSA_NLoS_var, ZSA_O2I_mean, ZSA_O2I_var] = ZSA_para_gen(fc,kase)
% generate ZoA Spread parameters (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase: {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor


    switch lower(kase)
        case 'umi'
            ZSA_LoS_mean = -0.1*log10(1+fc)+0.73;
            ZSA_LoS_var = -0.04*log10(1+fc)+0.34;
            ZSA_NLoS_mean = -0.04*log10(1+fc)+0.92;
            ZSA_NLoS_var =-0.07*log10(1+fc)+0.41;
            ZSA_O2I_mean = 1.01;
            ZSA_O2I_var = 0.43;
        case 'uma'
            ZSA_LoS_mean =0.95;
            ZSA_LoS_var = 0.16;
            ZSA_NLoS_mean = -0.3236*log10(fc)+1.512;
            ZSA_NLoS_var =0.16;
            ZSA_O2I_mean = 1.01;
            ZSA_O2I_var = 0.43;
        case 'rma'
            ZSA_LoS_mean = 0.47;
            ZSA_LoS_var = 0.4;
            ZSA_NLoS_mean = 0.58;
            ZSA_NLoS_var = 0.37;
            ZSA_O2I_mean = 0.93;
            ZSA_O2I_var = 0.22;
        case 'inoff'
            ZSA_LoS_mean = -0.26*log10(1+fc)+1.44;
            ZSA_LoS_var = -0.04*log10(1+fc)+0.264;
            ZSA_NLoS_mean = -0.15*log10(1+fc)+1.387;
            ZSA_NLoS_var = -0.09*log10(1+fc)+0.746;
            ZSA_O2I_mean = NaN;
            ZSA_O2I_var = NaN;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            ZSA_LoS_mean = -0.2*log10(1+fc)+1.5;
            ZSA_LoS_var = 0.35;
            ZSA_NLoS_mean = -0.13*log10(1+fc)+1.45;
            ZSA_NLoS_var = 0.45;
            ZSA_O2I_mean = NaN;
            ZSA_O2I_var = NaN;
    end
end