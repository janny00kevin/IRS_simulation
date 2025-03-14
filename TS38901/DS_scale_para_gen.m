function [r_SF_var_LoS,r_SF_var_NLoS,r_SF_var_O2I] = DS_scale_para_gen(kase)
% generate r_tau delay distribution proportionality factor (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase:  {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor


    switch lower(kase)
        case 'umi'
            r_SF_var_LoS = 3;
            r_SF_var_NLoS = 2.1;
            r_SF_var_O2I = 2.2;
        case 'uma'
            r_SF_var_LoS = 2.5;
            r_SF_var_NLoS = 2.3;
            r_SF_var_O2I = 2.2;
        case 'rma'
            r_SF_var_LoS = 3.8;
            r_SF_var_NLoS = 1.7;
            r_SF_var_O2I = 1.7;
        case 'inoff'
            r_SF_var_LoS = 3.6;
            r_SF_var_NLoS = 3;
            r_SF_var_O2I = NaN;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            r_SF_var_LoS = 2.7;
            r_SF_var_NLoS = 3;
            r_SF_var_O2I = NaN;
    end
end