function [loc_BSs,loc_IRSs,loc_UEs,ori_IRS] = loc_gen(nBS,nIRSp,nUE,ang_offset,r1,r2,r3)
ang_BS = 0:2*pi/nBS:2*pi;
ang_BS = ang_BS(1:nBS);
ang_BS = ang_BS + (ang_offset/360*2*pi);
ori_BS = [cos(ang_BS+pi); sin(ang_BS+pi) ];
loc_BSs = (r1+r2+r3)*[cos(ang_BS); sin(ang_BS) ];

%%
ang_IRS = 0:pi/nIRSp*2:pi;
ang_IRS = ang_IRS(1:nIRSp/2);
ang_IRS = [ang_IRS;ang_IRS+pi];
ang_IRS = ang_IRS(:);
% ori_IRS_z = ang_IRS+ reshape(repmat([0.5*pi -0.5*pi], nIRSp/2, 1)', nIRSp, 1);
ori_IRS_z = ang_IRS+ 0.5*pi*ones(size(ang_IRS));
ori_IRS_z = [cos(ori_IRS_z'); sin(ori_IRS_z')];
ori_IRS_x = [cos(ang_IRS) sin(ang_IRS)]';
loc_IRSs = (r2+r3)*[cos(ang_IRS) sin(ang_IRS)]';
%%
ang_UE = 2*pi*rand(nUE,1);
rad_UE = r3*rand(nUE,1);
loc_UEs = [rad_UE.*cos(ang_UE) rad_UE.*sin(ang_UE)]';
%%
figure();
scatter(loc_BSs(1,:),loc_BSs(2,:),'ro');
hold on
quiver(loc_BSs(1,:),loc_BSs(2,:),ori_BS(1,:),ori_BS(2,:),0.2);
scatter(loc_IRSs(1,:),loc_IRSs(2,:),'bo');
quiver(loc_IRSs(1,:),loc_IRSs(2,:),ori_IRS_z(1,:),ori_IRS_z(2,:),0.3);
scatter(loc_UEs(1,:),loc_UEs(2,:),'go');
axis square 
grid on

%%
loc_BSs = [loc_BSs' 10*ones(nBS,1)]';
loc_UEs = [loc_UEs' 2*ones(nUE,1)]';
loc_IRSs = [loc_IRSs' 10*ones(nIRSp,1)]';
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