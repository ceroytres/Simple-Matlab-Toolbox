function h=gabor_kernel(kSize,theta,k,sig)

[x,y]=meshgrid(-fix(kSize(1)/2):fix(kSize(1)/2),...
    -fix(kSize(2)/2):fix(kSize(2)/2));

h=exp(j*k*(cos(theta)*x+sin(theta)*y))...
    .*exp(-.5*(k/sig)^2*(x.^2+y.^2));

end