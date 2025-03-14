% LOS probability
function prob = LOS_prob(kase,d_2d,h_UT,h_BS,r_clutter,h_clutter)
%% function Description
%{
This is a function to compute the LOS probability from TR 38.901 (table 7.4.2-1)
INPUT:
    kase(= case) :  {mustBeMember(kase,["RMa","UMa","UMi","InOff","InF-SL","InF-DL","InF-SH","InF-DH","InF-HH"])}
    d_2d: outdoor/indoor 2D distance between BS and UT (outdoor: uma/umi/rma, indoor: inOff/InF)
    h_UT(for uMa and InF) : height of UT
    h_BS(for InF) : height of BS
    r_clutter(only needed for InF): clutter density, between 0 and 1
    h_clutter(only needed for InF): effective clutter height, 0~10, <= ceiling
OUTPUT:
 prob : LOS probability between 0 and 1

%}
% 
    switch lower(kase)
        case 'rma'
            if d_2d <=10
                prob = 1;
            else
                prob = exp(-(d_2d-10)/1000);
            end
        case 'umi'
            if d_2d <= 18
                prob = 1;
            else
                prob = 18/d_2d + exp(-d_2d/36)*(1-18/d_2d);
            end
        case 'uma'
            if d_2d <= 18
                prob = 1;
            else
                if(h_UT<=13)
                    C_UT = 0;
                else
                    C_UT = ((h_UT-13)/10)^(1.5);
                end
                prob =(18/d_2d+exp(-d_2d/63)*(1-18/d_2d))*(1+C_UT*1.25*(d_2d/100)^3*exp(-d_2d/150));
            end
        case 'inoff' % use open office here
            % open office
            if (d_2d <= 5)
                prob = 1;
            elseif (d_2d <=49)
                prob = exp(-(d_2d-5)/70);
            else
                prob = exp(-(d_2d-49)/211.7)*0.54;
            end
%             % mixed office
%             if (d_2d <= 1.2)
%                 prob = 1;
%             elseif (d_2d <=6.5)
%                 prob = exp(-(d_2d-1.2)/4.7);
%             else
%                 prob = exp(-(d_2d-6.5)/32.6)*0.32;
%             end
        case {'inf-sl','inf-dl','inf-sh','inf-dh'}
            if(h_UT > h_clutter)
                warning('The UT hieght is larger than the clutter effective height. Consider use the InF-HH scenario instead')
            end
            switch lower(kase)
                case {'inf-sl','inf-dl'}
                    if (h_BS > h_clutter)
                        warning('The BS is higher than the clutter effective height. Consider use InF-xH scenarios instead')
                    end
                case {'inf-sh','inf-dh'}
                    if (h_BS < h_clutter)
                        warning('The BS is lower than the clutter effective height. Consider use InF-xL scenarios instead')
                    end
            end

            switch lower(kase)
                case {'inf-sl','inf-sh'}
                    if (r_clutter > 0.4)
                        warning('The clutter density is greater than 0.4. Consider use InF-Dx scenarios instead')
                    end
                case {'inf-dl','inf-dh'}
                    if (r_clutter < 0.4)
                        warning('The clutter density is lesser than 0.4. Consider use InF-Sx scenarios instead')
                    end
            end
            
            switch lower(kase)
            case 'inf-sl'
                d_clutter = 10;
            case 'inf-sh'
                d_clutter = 2;
            case 'inf-dl'
                d_clutter = 10;
            case 'inf-dh'
                d_clutter = 2;
            end

            switch lower(kase)
                case {'inf-sl','inf-dl'}
                    k_subsce = -d_clutter/log(1-r_clutter);
                case {'inf-sh','inf-dh'}
                    k_subsce = -d_clutter/log(1-r_clutter)* (h_BS-h_UT)/(h_clutter-h_UT);
            end
            prob = exp(-d_2d/k_subsce);
        case 'inf-hh'
            prob = 1;



    end
end