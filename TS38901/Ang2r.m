function [r] = Ang2r(Theta,Phi)
    % theta and Phi in degree
    r =[ sind(Theta)*cosd(Phi) sind(Theta)*sind(Phi) cosd(Theta)].';
end