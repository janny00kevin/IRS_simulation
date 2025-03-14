function sig_vec = func_sig_vec(theta,n,delta)
%theta   = (60/180)*pi;
%n       =   4;

index   =   [0:(n-1)]*delta;  
index   =   index.';
sig_vec =   exp(j*pi*sin(theta)*index);