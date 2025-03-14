function R = LCS2GCS_xyz(alph,betta,gamma)
% matches 2 different(local and global) coordination system by shift the x axis for alph, y axis for beta and z axis for gamma
% R^(-1) = R^T (i.e. GCS 2 LCS)
    r11 = cos(alph)*cos(betta);
    r12 = cos(alph)*sin(betta)*sin(gamma)-sin(alph)*cos(gamma);
    r13 = cos(alph)*sin(betta)*cos(gamma)+sin(alph)*sin(gamma);
    r21 = sin(alph)*cos(betta);
    r22 = sin(alph)*sin(betta)*sin(gamma)+cos(alph)*cos(gamma);
    r23 = sin(alph)*sin(betta)*cos(gamma)-cos(alph)*sin(gamma);
    r31 = -sin(betta);
    r32 = cos(betta)*sin(gamma);
    r33 = cos(betta)*cos(gamma);
    R = [r11 r12 r13; r21 r22 r23;r31 r32 r33];
end

% function [theta_L,phi_L] = GCS2LCS_angs(alph,betta,gamma,theta,phi)
%     theta_L = 
% end
