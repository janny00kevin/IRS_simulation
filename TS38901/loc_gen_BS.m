function [loc_BSs,loc_IRSs,ori_IRS] = loc_gen_BS(nBS,nIRSp,ang_offset,r1,r2,r3,h_BS,h_IRS,DOUBLESIDE)
%     Compute the location of BS and IRS (and randomly generate locations of
%     UE in a circle of the radius r1+r2+r3)
%     BSs are evenly placed on the line segment (0,k) to (0,d_y) 
%     IRSs are evenly placed on the line segment (d_BS_IRS,0) to (d_BS_IRS,d_y-k) 
%     UEs are randomly placed on in the retangle [(d_BS_IRS+d_IRS_UE,0),(d_BS_IRS+d_IRS_UE,d_y),(d_BS_IRS+d_IRS_UE+d_UE,0),(d_BS_IRS+d_IRS_UE+d_UE,d_y)] 
%     nBS,nUE,nIRS : number of BS/UE/IRS
%     h_BS/h_UE: height of BS(IRS)/UE
%     loc_BS/loc_UE/loc_IRS: location of BS/UE/IRS in cartesian coordinate,
%     3-by-nUE/nBS/nIRS
%     ori_IRS: unit vector along the normal  (defined as the normal direction)

        ang_BS = 0:2*pi/nBS:2*pi;
        ang_BS = ang_BS(1:nBS);
        ang_BS = ang_BS + (ang_offset/360*2*pi);
        ori_BS = [cos(ang_BS+pi); sin(ang_BS+pi) ];
        loc_BSs = (r1+r2+r3)*[cos(ang_BS); sin(ang_BS) ];
        
        %%
        
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
        
        ang_IRS = ang_IRS(:);
        if(DOUBLESIDE)
            ori_IRS_z = ang_IRS+ reshape(repmat([0.5*pi -0.5*pi], nIRSp/2, 1)', nIRSp, 1);%double side
        else
            ori_IRS_z = ang_IRS+ 0.5*pi*ones(size(ang_IRS)); %single-side
        end
        
        
        ori_IRS_z = [cos(ori_IRS_z'); sin(ori_IRS_z')];
        ori_IRS_x = [cos(ang_IRS) sin(ang_IRS)]';
        loc_IRSs = (r2+r3)*[cos(ang_IRS) sin(ang_IRS)]';
        %%
        if 1
            figure();
            scatter(loc_BSs(1,:),loc_BSs(2,:),'ro');
            hold on
            quiver(loc_BSs(1,:),loc_BSs(2,:),ori_BS(1,:),ori_BS(2,:),0.2);
            scatter(loc_IRSs(1,:),loc_IRSs(2,:),'bo');
            quiver(loc_IRSs(1,:),loc_IRSs(2,:),ori_IRS_z(1,:),ori_IRS_z(2,:),0.3);
            % scatter(loc_UEs(1,:),loc_UEs(2,:),'go');
            grid on
            axis square 

            close;
        end
        %%
        loc_BSs = [loc_BSs' h_BS*ones(nBS,1)]';
        loc_IRSs = [loc_IRSs' h_IRS*ones(nIRSp,1)]';
        % ori_BS = [ori_BS' zeros(nBS,1)];
        ori_IRS_z = [ori_IRS_z' zeros(nIRSp,1)]';
        ori_IRS_x = [ori_IRS_x' zeros(nIRSp,1)]';
        
        ori_IRS = zeros(3,3,nIRSp);
        for idxIRS = 1:nIRSp
            ori_IRS(:,1,idxIRS) = ori_IRS_x(:,idxIRS);
            ori_IRS(:,2,idxIRS) = [0 0 1];
            ori_IRS(:,3,idxIRS) = ori_IRS_z(:,idxIRS);
        end
end