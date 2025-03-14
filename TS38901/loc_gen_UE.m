function [loc_UEs] = loc_gen_UE(nUE,r3,h_UE,NoDeadZone,b_IRS_access,DOUBLESIDE,loc_IRS,ori_IRS)
% b_IRS_access:  boolean vector indicate the IRS this AP has access to. 1
% when the IRS can be reached
if(NoDeadZone)
    % find the feasible range of angles
    nIRSp = length(b_IRS_access);
    if(DOUBLESIDE)
        ang_IRS = 0:pi/nIRSp*2:pi;
        ang_IRS = ang_IRS(1:nIRSp/2);
        ang_IRS = 2*ang_IRS;
        ang_IRS = [ang_IRS;ang_IRS]; %double side
    else
        ang_IRS = 0:pi/nIRSp*2:pi;
        ang_IRS = ang_IRS(1:nIRSp/2);
        ang_IRS = [ang_IRS;ang_IRS+pi]; %single-side
    end

    ang_IRS = ang_IRS(b_IRS_access==true);

    if(DOUBLESIDE)
        ori_IRS_LB = ang_IRS+ reshape(repmat([0 pi], nIRSp/2, 1)', nIRSp, 1);
        ori_IRS_UB = ang_IRS+ reshape(repmat([pi 2*pi], nIRSp/2, 1)', nIRSp, 1);%double side
    else
        ori_IRS_LB = ang_IRS; %single-side
        ori_IRS_UB = ang_IRS+ pi*ones(size(ang_IRS));
    end
    
    % make angles between 0~2pi
    ori_IRS_LB(ori_IRS_LB~=0) = ori_IRS_LB(ori_IRS_LB~=0)-(floor(ori_IRS_LB(ori_IRS_LB~=0)/2/pi)+1*2*pi);
    ori_IRS_UB(ori_IRS_UB~=2*pi) = ori_IRS_UB(ori_IRS_UB~=2*pi)-floor(ori_IRS_UB(ori_IRS_UB~=2*pi)/2/pi)*2*pi;
    
    if(min(ori_IRS_LB)<=-pi)
        ori_IRS_LB(ori_IRS_LB==0) = -2*pi;
    end
    
    if(min(ori_IRS_UB)<=pi)
        ori_IRS_UB(ori_IRS_UB==2*pi) = 0;
    end
    
    ori_IRS_LB_t = min(ori_IRS_LB);
    ori_IRS_UB_t = max(ori_IRS_UB); 
    ori_IRS_UB_t = ori_IRS_UB_t-floor((ori_IRS_UB_t-ori_IRS_LB_t)/2/pi)*2*pi;


    % randomly place UEs
    ang_UE = ori_IRS_LB_t+(ori_IRS_UB_t-ori_IRS_LB_t)*rand(nUE,1);
    rad_UE = r3*rand(nUE,1);
    loc_UEs = [rad_UE.*cos(ang_UE) rad_UE.*sin(ang_UE)]';
    loc_UEs = [loc_UEs' h_UE*ones(nUE,1)]';
    
    % double check
    loc_IRS = loc_IRS(:,b_IRS_access==true);
    ori_IRS = ori_IRS(:,:,b_IRS_access==true);
    nIRSp_eff = size(loc_IRS,2);
    f_disable = ones(nIRSp_eff,nUE);
    for idxUE = 1:nUE
        loc_UE_curr = loc_UEs(:,idxUE);
        for idxIRS = 1:nIRSp_eff
            loc_IRS_curr = loc_IRS(:,idxIRS);
            if((loc_UE_curr-loc_IRS_curr).'*ori_IRS(:,3,idxIRS)<0)
                f_disable(idxIRS,idxUE) = 0;
            end
        end
    end
    f_disable = sum(f_disable,1);
    if(sum(f_disable==0))
        disp('double check failed!');
    end
else
%%
    ang_UE = 2*pi*rand(nUE,1);
    rad_UE = r3*rand(nUE,1);
    loc_UEs = [rad_UE.*cos(ang_UE) rad_UE.*sin(ang_UE)]';
    %%
    loc_UEs = [loc_UEs' h_UE*ones(nUE,1)]';

end
end