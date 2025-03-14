function [ASA_LoS_mean,ASA_LoS_var,ASA_NLoS_mean,ASA_NLoS_var, ASA_O2I_mean, ASA_O2I_var] = ASA_para_gen(fc,kase)
% generate AoA Spread parameters (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase: {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor


    switch lower(kase)
        case 'umi'
            ASA_LoS_mean = -0.08*log10(1+fc)+1.73;
            ASA_LoS_var = 0.014*log10(1+fc)+0.28;
            ASA_NLoS_mean = -0.08*log10(1+fc)+1.81;
            ASA_NLoS_var =0.05*log10(1+fc)+1.21;
            ASA_O2I_mean = 1.76;
            ASA_O2I_var = 0.16;
        case 'uma'
            ASA_LoS_mean =1.81;
            ASA_LoS_var = 0.20;
            ASA_NLoS_mean = 2.08-0.27*log10(fc);
            ASA_NLoS_var =0.11;
            ASA_O2I_mean = 1.76;
            ASA_O2I_var = 0.16;
        case 'rma'
            ASA_LoS_mean = 1.52;
            ASA_LoS_var = 0.24;
            ASA_NLoS_mean = 1.52;
            ASA_NLoS_var =0.13;
            ASA_O2I_mean = 1.66;
            ASA_O2I_var = 0.21;
        case 'inoff'
            ASA_LoS_mean = -0.19*log10(1+fc)+1.781;
            ASA_LoS_var = 0.12*log10(1+fc)+0.119;
            ASA_NLoS_mean = -0.11*log10(1+fc)+1.863;
            ASA_NLoS_var =0.12*log10(1+fc)+0.059;
            ASA_O2I_mean = NaN;
            ASA_O2I_var = NaN;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            ASA_LoS_mean = -0.18*log10(1+fc)+1.78;
            ASA_LoS_var = 0.12*log10(1+fc)+0.2;
            ASA_NLoS_mean = 1.72;
            ASA_NLoS_var =0.3;
            ASA_O2I_mean = NaN;
            ASA_O2I_var = NaN;
    end
end