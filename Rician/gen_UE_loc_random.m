function [location_UE]= gen_UE_loc_random(area_UE_center,area_UE_x,area_UE_y,num_UE,num_MC)

% x = (area_UE_center - area_UE_x) ~ (area_UE_center + area_UE_x)
%y = -area_UE_y ~ area_UE_y

location_UE = zeros(3,num_UE);

for idx_MC = 1:num_MC
    for idx_UE = 1:num_UE
        location_UE(1,idx_UE,idx_MC) = area_UE_center + randi([-area_UE_x, area_UE_x],1,1);
        location_UE(2,idx_UE,idx_MC) = randi([-area_UE_y, area_UE_y],1,1);
    end
end


end