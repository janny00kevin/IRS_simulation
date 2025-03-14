function out = str_ULA(theta,dx,nx)
% dx,dy: inter element distance
% nx.ny: num. of elements along x/y axis
% angles in degree
x = 0:nx-1;
out = exp(-1j*pi*x*dx*sind(theta)).';
end