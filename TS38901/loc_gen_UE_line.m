function [loc_UE] = loc_gen_UE_line(nUE,d_BS_IRS,d_IRS_UE,d_UE,d_y,h_UE)
%     Compute the location of BS and IRS and randomly generate locations of
%     UE in a rectangle of size d_BS_IRS+d_IRS_UE+d_UE by d_y
%     BSs are evenly placed on the line segment (0,k) to (0,d_y) 
%     IRSs are evenly placed on the line segment (d_BS_IRS,0) to (d_BS_IRS,d_y-k) 
%     UEs are randomly placed on in the retangle [(d_BS_IRS+d_IRS_UE,0),(d_BS_IRS+d_IRS_UE,d_y),(d_BS_IRS+d_IRS_UE+d_UE,0),(d_BS_IRS+d_IRS_UE+d_UE,d_y)] 
%     nBS,nUE,nIRS : number of BS/UE/IRS
%     h_BS/h_UE: height of BS(IRS)/UE
%     loc_BS/loc_UE/loc_IRS: location of BS/UE/IRS in cartesian coordinate,
%     3-by-nUE/nBS/nIRS


%     loc_BS = zeros(3,nBS);
    loc_UE = zeros(3,nUE);
%     loc_IRS = zeros(3,nIRS);
%     loc_BS(3,:) = h_BS;
    loc_UE(3,:) = h_UE;
%     loc_IRS(3,:) = h_BS;
    
%     loc_BS(1,:) = 0;
%     loc_IRS(1,:) = d_BS_IRS;
    loc_UE(1,:) = d_BS_IRS+d_IRS_UE+d_UE*rand(1,nUE);
    loc_UE(2,:) = d_y*rand(1,nUE);

%     offset = d_y/10;
%     d_BS = (d_y-offset)/(nBS-1);
%     loc_BS(2,:) = offset+d_BS*(0:nBS-1);
%     if nIRS == 1
%         loc_IRS(2,:) = 0;
%     else
%         d_IRS = (d_y-offset)/(nIRS-1);
%         loc_IRS(2,:) = d_IRS*(0:nIRS-1);
%     end
%     ori_IRS = ones(nIRS,1);

    if 0
        figure;
        plot(loc_IRS(1,:),loc_IRS(2,:),'go');
        hold on;
        plot(loc_BS(1,:),loc_BS(2,:),'b^');
        plot(loc_UE(1,:),loc_UE(2,:),'rx');       
    
    end
    
    
end