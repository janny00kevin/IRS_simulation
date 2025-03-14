function out = str_UPA(phi,theta,dx,dy,nx,ny)
% dx,dy: inter element distance
% nx.ny: num. of elements along x/y axis
% angles in degree
x = reshape(repmat(0:nx-1, ny, 1)', nx*ny, 1);
y = reshape(repmat(0:ny-1, nx, 1), nx*ny, 1);
out = exp(1j*pi*x*dx*sind(theta)*cosd(phi)).*exp(1j*pi*dy*y*sind(theta)*sind(phi));
end