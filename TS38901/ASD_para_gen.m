function [ASD_LoS_mean,ASD_LoS_var,ASD_NLoS_mean,ASD_NLoS_var, ASD_O2I_mean, ASD_O2I_var] = ASD_para_gen(fc,kase)
% generate AoD Spread parameters (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase: {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor


    switch lower(kase)
        case 'umi'
            ASD_LoS_mean = -0.05*log10(1+fc)+1.21;
            ASD_LoS_var = 0.41;
            ASD_NLoS_mean = -0.23*log10(1+fc)+1.53;
            ASD_NLoS_var =0.11*log10(1+fc)+0.33;
            ASD_O2I_mean = 1.25;
            ASD_O2I_var = 0.42;
        case 'uma'
            ASD_LoS_mean = 1.06+0.1114*log10(fc);
            ASD_LoS_var = 0.28;
            ASD_NLoS_mean = 1.5-0.1144*log10(fc);
            ASD_NLoS_var =0.28;
            ASD_O2I_mean = 1.25;
            ASD_O2I_var = 0.42;
        case 'rma'
            ASD_LoS_mean = 0.9;
            ASD_LoS_var = 0.38;
            ASD_NLoS_mean = 0.95;
            ASD_NLoS_var =0.45;
            ASD_O2I_mean = -0.67;
            ASD_O2I_var = 0.17;
        case 'inoff'
            ASD_LoS_mean = 1.6;
            ASD_LoS_var = 0.18;
            ASD_NLoS_mean = 1.62;
            ASD_NLoS_var =0.25;
            ASD_O2I_mean = NaN;
            ASD_O2I_var = NaN;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            ASD_LoS_mean = 1.56;
            ASD_LoS_var = 0.25;
            ASD_NLoS_mean = 1.57;
            ASD_NLoS_var =0.2;
            ASD_O2I_mean = NaN;
            ASD_O2I_var = NaN;
    end
end