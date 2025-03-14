function [rad,azi,eva] = xyz2pol(car_t)
% car_t: 3x1
    rad =  norm(car_t);
    unit = car_t/rad;
    eva = acosd(unit(3)); % degrees
    azi = 180*angle([1 1j 0]*unit)/pi; % degrees
end