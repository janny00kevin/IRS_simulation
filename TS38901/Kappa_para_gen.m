function [Kappa_mean,Kappa_std] = Kappa_para_gen(kase)
% generate ZoA Spread parameters (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase:  {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor


    switch lower(kase)
        case 'umi'
            Kappa_mean = 9;
            Kappa_std = 5;
        case 'uma'
            Kappa_mean = 9;
            Kappa_std = 3.5;
        case 'rma'
            Kappa_mean = 7;
            Kappa_std = 4;
        case 'inoff'
            Kappa_mean = 7;
            Kappa_std = 4;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            Kappa_mean = 7;
            Kappa_std = 8;
    
    end
end