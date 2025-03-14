function [loc_BS,loc_IRS,ori_IRS] = loc_gen_BS_line(nBS,nIRSp,d_BS_IRS_L,d_y,h_BS,h_IRS)
%     Compute the location of BS and IRS (and randomly generate locations of
%     UE in a circle of the radius r1+r2+r3)
%     BSs are evenly placed on the line segment (0,k) to (0,d_y) 
%     IRSs are evenly placed on the line segment (d_BS_IRS,0) to (d_BS_IRS,d_y-k) 
%     UEs are randomly placed on in the retangle [(d_BS_IRS+d_IRS_UE,0),(d_BS_IRS+d_IRS_UE,d_y),(d_BS_IRS+d_IRS_UE+d_UE,0),(d_BS_IRS+d_IRS_UE+d_UE,d_y)] 
%     nBS,nUE,nIRS : number of BS/UE/IRS
%     h_BS/h_UE: height of BS(IRS)/UE
%     loc_BS/loc_UE/loc_IRS: location of BS/UE/IRS in cartesian coordinate,
%     3-by-nUE/nBS/nIRS
%     ori_IRS: unit vector along the normal  (basis vector of 3D, z axis defined as the normal direction)

    loc_BS = zeros(3,nBS);
%     loc_UE = zeros(3,nUE);
    loc_IRS = zeros(3,nIRSp);
    ori_IRS = zeros(3,3,nIRSp);
    loc_BS(3,:) = h_BS;
%     loc_UE(3,:) = h_UE;
    loc_IRS(3,:) = h_IRS;
    
    loc_BS(1,:) = 0;
    loc_IRS(1,:) = d_BS_IRS_L;
%     loc_UE(1,:) = d_BS_IRS+d_IRS_UE+d_UE*rand(1,nUE);
%     loc_UE(2,:) = d_y*rand(1,nUE);

%     offset = d_y/10;
    if(nBS==1)
        loc_BS(2,:) = d_y;
    else
        d_BS = d_y/(nBS-1);
        loc_BS(2,:) = d_BS*(0:nBS-1);
    end
    
    nIRS_loc = ceil(nIRSp/2);
    
    if nIRSp == 1
        loc_IRS(2,1) = 0;
    elseif nIRSp == 2
        loc_IRS(2,1) = 0;
        loc_IRS(2,2) = d_y;
    else
        loc_IRS(2,1) = 0;
        loc_IRS(2,2) = d_y;
        d_IRS = (d_y)/(nIRS_loc);
        loc_IRS(2,3:2:end) = d_IRS*(1:nIRS_loc-1);
        if mod(nIRSp,2) == 1
            loc_IRS(2,4:2:end) = d_IRS*(1:nIRS_loc-2);
        else
            loc_IRS(2,4:2:end) = d_IRS*(1:nIRS_loc-1);
        end
    end
%     ori_IRS = ones(nIRSp,1);
%     ori_IRS(2:2:end) = -1;
    for idxIRS = 1:nIRSp
        if(mod(idxIRS,2)==1)
            ori_IRS(:,:,idxIRS) = [1 0 0;0 0 1;0 1 0];
        else
            ori_IRS(:,:,idxIRS) = [-1 0 0;0 0 -1;0 1 0];
        end
    end
    
    if 0
        ori_IRS_z = squeeze(ori_IRS(:,3,:));


        figure();
        scatter(loc_BS(1,:),loc_BS(2,:),'ro');
        hold on
%         quiver(loc_BSs(1,:),loc_BSs(2,:),ori_BS(1,:),ori_BS(2,:),0.2);
        scatter(loc_IRS(1,:),loc_IRS(2,:),'bo');
        quiver(loc_IRS(1,:),loc_IRS(2,:),ori_IRS_z(1,:),ori_IRS_z(2,:),0.3);
        % scatter(loc_UEs(1,:),loc_UEs(2,:),'go');
        grid on
        axis square

    end

end