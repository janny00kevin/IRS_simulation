function [Clu_SF_var_LoS,Clu_SF_var_NLoS,Clu_SF_var_O2I] = Clu_SF_para_gen(kase)
% generate Per cluster shadowing std in dB (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase: {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor


    switch lower(kase)
        case 'umi'
            Clu_SF_var_LoS = 3;
            Clu_SF_var_NLoS = 3;
            Clu_SF_var_O2I = 4;
        case 'uma'
            Clu_SF_var_LoS = 3;
            Clu_SF_var_NLoS = 3;
            Clu_SF_var_O2I = 4;
        case 'rma'
            Clu_SF_var_LoS = 3;
            Clu_SF_var_NLoS = 3;
            Clu_SF_var_O2I = 3;
        case 'inoff'
            Clu_SF_var_LoS = 6;
            Clu_SF_var_NLoS = 3;
            Clu_SF_var_O2I = NaN;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            Clu_SF_var_LoS = 4;
            Clu_SF_var_NLoS = 3;
            Clu_SF_var_O2I = NaN;
    end
end