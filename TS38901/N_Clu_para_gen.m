function [nClu_LoS,nRay_LoS,nClu_NLoS,nRay_NLoS, nClu_O2I, nRay_O2I] = N_Clu_para_gen(kase)
% generate number of clusters/ ray per cluster (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase:  {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor


    switch lower(kase)
        case 'umi'
            nClu_LoS = 12;
            nRay_LoS = 20;
            nClu_NLoS = 19;
            nRay_NLoS = 20;
            nClu_O2I = 12;
            nRay_O2I = 20;
        case 'uma'
            nClu_LoS = 12;
            nRay_LoS = 20;
            nClu_NLoS = 20;
            nRay_NLoS = 20;
            nClu_O2I = 12;
            nRay_O2I = 20;
        case 'rma'
            nClu_LoS = 11;
            nRay_LoS = 20;
            nClu_NLoS = 10;
            nRay_NLoS = 20;
            nClu_O2I = 10;
            nRay_O2I = 20;
        case 'inoff'
            nClu_LoS = 15;
            nRay_LoS = 20;
            nClu_NLoS = 19;
            nRay_NLoS = 20;
            nClu_O2I = NaN;
            nRay_O2I = NaN;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            nClu_LoS = 25;
            nRay_LoS = 20;
            nClu_NLoS = 25;
            nRay_NLoS = 20;
            nClu_O2I = NaN;
            nRay_O2I = NaN;
    end
end