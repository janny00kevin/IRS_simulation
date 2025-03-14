function [PL_NLOS,PL_LOS,sig_SF_NLOS,sig_SF_LOS] = cal_PL(loc_BS,loc_UT,scenario,f_c)
% calculating pathloss (Table 7.4.1-1, use the default for NLOS (not the optional one))
% use the complicated formula for PL_NLOS (i.e. PL_NLOS depends on PL_LOS)
%INPUTS:
% loc_BS, loc_UT: location of BS/UT in xyz coordinate, 3 by 1 (in m)
% scenario: {mustBeMember(scenario,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
% f_c: central frequency (in GHz)
%OUTPUTS:
%PL : pathlos in dB
%sig_SF: shadow fading std in dB


d_2d = norm(loc_BS(1:2)-loc_UT(1:2));
d_3d = norm(loc_BS-loc_UT);
switch lower(scenario)
    case 'rma'
        h_avg = 5; w_avg = 5; % default value, average building height and street width
        d_BP = 2*pi*loc_BS(3)*loc_UT(3)*f_c/0.3; %break point distance
        PL_LOS =  20*log10(40*pi*f_c/3)+min(0.03*h_avg^1.72,10)*log10(d_3d)...
            -min(0.044*h_avg^1.72,14.77)+0.002*log10(h_avg)*d_3d;
        sig_SF_LOS = 4;
        if d_2d >= d_BP
            PL_LOS = PL_LOS + 40*log10(d_3d/d_BP);
            sig_SF_LOS = 6;
        end
        
        PL_NLOS = 161.04-7.1*log10(w_avg)+7.5*log10(h_avg)-(24.37-3.7*(h_avg/loc_BS(3))^2)*log10(loc_BS(3))...
            +(43.42-3.1*log10(loc_BS(3)))*(log10(d_3d)-3)...
            +20*log10(f_c)-(3.2*(log10(11.75*loc_UT(3)))^2-4.97);
        PL_NLOS = max(PL_NLOS,PL_LOS);
        sig_SF_NLOS = 8;
    case 'umi'
        h_E = 1; %effective environment height
        d_BP = 4*(loc_BS(3)-h_E)*(loc_UT(3)-h_E)*f_c/0.3;
        sig_SF_LOS = 4;
        sig_SF_NLOS = 7.82;
        
        if d_2d <= d_BP
            PL_LOS = 32.4+21*log10(d_3d)+20*log10(f_c);
        else
            PL_LOS = 32.4+40*log10(d_3d)+20*log10(f_c)-9.5*log10(d_BP^2+(loc_BS(3)-loc_UT(3))^2);            
        end
        
        PL_NLOS = 22.4+35.3*log10(d_3d)+21.3*log10(f_c)-0.3*(loc_UT(3)-1.5);
        PL_NLOS = max(PL_NLOS,PL_LOS);
        
    case 'uma'
        if d_2d <= 18
            g = 0;
        else
            g = 1.25*(d_2d/100)^100*exp(-d_2d/150);
        end
        
        if loc_UT(3)<=13
            c = 0;
        else
            c = ((loc_UT(3)-13)/10)^1.5*g;
        end
        
        if(rand(1)<=(1/(1+c)))
            h_E = 1; %effective environment height
        else
            sample = 12:3:(loc_UT(3)-1.5);
            h_E = sample(randi(length(sample),1)); %effective environment height
        end
        
        
        
        d_BP = 4*(loc_BS(3)-h_E)*(loc_UT(3)-h_E)*f_c/0.3;
        sig_SF_LOS = 4;
        sig_SF_NLOS = 6;
        
        if d_2d <= d_BP
            PL_LOS = 28+22*log10(d_3d)+20*log10(f_c);
        else
            PL_LOS = 28+40*log10(d_3d)+20*log10(f_c)-9*log10(d_BP^2+(loc_BS(3)-loc_UT(3))^2);            
        end
        
        PL_NLOS = 13.54+39.08*log10(d_3d)+20*log10(f_c)-0.6*(loc_UT(3)-1.5);
        PL_NLOS = max(PL_NLOS,PL_LOS);
    case 'inoff' % d_3d <= 150m
        if(d_3d>150)
            warning('The BS-UT distance not suitible for the indoor office scenario. (d_3d <= 150m)')
        end
        PL_LOS= 32.4+17.3*log10(d_3d)+20*log10(f_c);
        PL_NLOS= 17.3+38.3*log10(d_3d)+24.9*log10(f_c);
        PL_NLOS = max(PL_NLOS,PL_LOS);
        sig_SF_NLOS= 8.03;
        sig_SF_LOS = 3;
 
    case {'inf-sl','inf-dl','inf-sh','inf-dh','inf-hh'} % d_3d <= 600m
        if(d_3d>600)
            warning('The BS-UT distance not suitible for the indoor office scenario. (d_3d <= 150m)')
        end
        PL_LOS= 31.84+21.5*log10(d_3d)+19*log10(f_c);
        sig_SF_LOS = 4;
        switch lower(scenario)
            case 'inf-sl'
                PL_NLOS= 33+25.5*log10(d_3d)+20*log10(f_c);
                PL_NLOS = max(PL_NLOS,PL_LOS);
                sig_SF_NLOS= 5.7;
            case 'inf-dl'
                PL_NLOS_SL= 33+25.5*log10(d_3d)+20*log10(f_c);
                PL_NLOS= 18.6+35.7*log10(d_3d)+20*log10(f_c);
                PL_NLOS = max([PL_NLOS,PL_LOS,PL_NLOS_SL]);
                sig_SF_NLOS= 7.2;
            case 'inf-sh'
                PL_NLOS= 32.4+23*log10(d_3d)+20*log10(f_c);
                PL_NLOS = max(PL_NLOS,PL_LOS);
                sig_SF_NLOS= 5.9;
            case 'inf-dh'
                PL_NLOS= 33.63+21.9*log10(d_3d)+20*log10(f_c);
                PL_NLOS = max(PL_NLOS,PL_LOS);
                sig_SF_NLOS= 4;
            case 'inf-hh' % not applicable (all LOS links)
                PL_NLOS= PL_LOS;
                sig_SF_NLOS= sig_SF_LOS;
        end
        
end

end