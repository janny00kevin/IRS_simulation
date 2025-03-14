function [DS_LoS_mean,DS_LoS_std,DS_NLoS_mean,DS_NLoS_std, DS_O2I_mean, DS_O2I_std] = DS_para_gen(fc,kase,room_size)
% generate delay spread parameters in dB (table 7.5-6)
% fc: central/carrier frequency in GHz
% kase: {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and the standard variation of the delay
% spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor
% room_size (only for InF): 1 by 3 [length width height] of the cube room
% (in m)


    switch lower(kase)
        case 'umi'
            DS_LoS_mean = -0.24*log10(1+fc)-7.14;
            DS_LoS_std = 0.38;
            DS_NLoS_mean = -0.24*log10(1+fc)-6.83;
            DS_NLoS_std =0.16*log10(1+fc)+0.28;
            DS_O2I_mean = -6.62;
            DS_O2I_std = 0.32;
        case 'uma'
            DS_LoS_mean = -6.955-0.0963*log10(fc);
            DS_LoS_std = 0.66;
            DS_NLoS_mean = -6.28-0.204*log10(fc);
            DS_NLoS_std =0.39;
            DS_O2I_mean = -6.62;
            DS_O2I_std = 0.32;
        case 'rma'
            DS_LoS_mean = -7.49;
            DS_LoS_std = 0.55;
            DS_NLoS_mean = -7.43;
            DS_NLoS_std =0.48;
            DS_O2I_mean = -7.47;
            DS_O2I_std = 0.24; 
        case 'inoff'
            DS_LoS_mean = -7.692-0.01*log10(1+fc);
            DS_LoS_std = 0.18;
            DS_NLoS_mean = -7.173-0.28*log10(1+fc);
            DS_NLoS_std =0.055+0.1*log10(1+fc);
            DS_O2I_mean = NaN;
            DS_O2I_std = NaN;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            V = room_size(1)*room_size(2)*room_size(3); %hall volume
            S = 2*(room_size(1)*room_size(2)+room_size(2)*room_size(3)+room_size(1)*room_size(3));
            %total surface area of hall in m2 (walls+floor+ceiling)
            DS_LoS_mean = log10(26*(V/S)+14)-9.35;
            DS_LoS_std = 0.15;
            DS_NLoS_mean = log10(30*(V/S)+32)-9.44;
            DS_NLoS_std = 0.19;
            DS_O2I_mean = NaN;
            DS_O2I_std = NaN;

    end
end