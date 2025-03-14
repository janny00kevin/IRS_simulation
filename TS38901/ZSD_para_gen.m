function [ZSD_LoS_mean,ZSD_LoS_var, ZSD_LoS_OS,ZSD_NLoS_mean,ZSD_NLoS_var, ZSD_NLoS_OS] = ZSD_para_gen(f_c,kase,loc_BS,loc_UT)
% generate ZoD Spread parameters (in dB)
% fc: central/carrier frequency in GHz
% kase: {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% LoS_mean,LoS_var,NLoS_mean,NLoS_var, O2I_mean, O2I_var: the mean and
% variance of the azimuth angular spread
% LoS: Line of sight
% NLoS: Non Line of Sight
% O2I: outdoor to indoor
%table 7.5-7~11
d_2d = norm(loc_BS(1:2)-loc_UT(1:2));

    switch lower(kase)
        case 'umi'
            ZSD_LoS_mean = max(-0.21,-14.8*(d_2d/1000)+0.01*abs(loc_UT(3)-loc_BS(3))+0.83);
            ZSD_LoS_var = 0.35;
            ZSD_NLoS_mean = max(-0.5,-3.1*(d_2d/1000)+0.01*max(loc_UT(3)-loc_BS(3),0)+0.2);
            ZSD_NLoS_var =0.35;
            ZSD_LoS_OS = 0;
            ZSD_NLoS_OS = -10^(-1.5*log10(max(10,d_2d)+3.3));
        case 'uma'
            ZSD_LoS_mean = max(-0.5,-2.1*(d_2d/1000)-0.01*(loc_UT(3)-1.5)+0.75);
            ZSD_LoS_var = 0.4;
            ZSD_NLoS_mean = max(-0.5,-2.1*(d_2d/1000)-0.01*(loc_UT(3)-1.5)+0.9);
            ZSD_NLoS_var =0.49;
            ZSD_LoS_OS = 0;
            ZSD_NLoS_OS = 7.66*log10(f_c)-5.96-10^((0.208*log10(f_c)-0.782)*log10(max(d_2d,25))-0.13*log10(f_c)+2.03-0.07*(loc_UT(3)-1.5));
        case 'rma' % not implemented/ including O2I
            ZSD_LoS_mean = max(-1,-0.17*(d_2d/1000)-0.01*(loc_UT(3)-1.5)+0.22);
            ZSD_LoS_var = 0.34;
            ZSD_NLoS_mean = max(-1,-0.19*(d_2d/1000)-0.01*(loc_UT(3)-1.5)+0.28);
            ZSD_NLoS_var =0.30;
            ZSD_LoS_OS = 0;
            ZSD_NLoS_OS = atan((35-3.5)/d_2d)-atan((35-1.5)/d_2d);
%             ZSD_O2I_mean = max(-1,-0.19*(d_2d/1000)-0.01*(loc_UT(3)-1.5)+0.28);
%             ZSD_O2I_var =0.30;
%             ZSD_O2I_OS = atan((35-3.5)/d_2d)-atan((35-1.5)/d_2d);
        case 'inoff'
            ZSD_LoS_mean = -1.43*log10(1+f_c)+2.228;
            ZSD_LoS_var = 0.13*log10(1+f_c)+0.3;
            ZSD_NLoS_mean = 1.08;
            ZSD_NLoS_var =0.36;
            ZSD_LoS_OS = 0;
            ZSD_NLoS_OS = 0;
        case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'}
            ZSD_LoS_mean = 1.35;
            ZSD_LoS_var = 0.35;
            ZSD_NLoS_mean = 1.2;
            ZSD_NLoS_var =0.55;
            ZSD_LoS_OS = 0;
            ZSD_NLoS_OS = 0;
    end
end