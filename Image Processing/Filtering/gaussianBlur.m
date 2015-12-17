function dst=gaussianBlur(src,sigma)
% dst=gaussianBlur(src,sigma)
% 
% Gaussian blurs an image
%
% Input:
%     src: grayscale image
%     sigma: sigma for smoothing
% Output:
%     dst: blurred image

s=ceil(2*sigma);
x=-s:s;

h=exp(-(x.^2)/(2*sigma^2));
h=h'*h;
h=h/sum(h(:));

dst=imfilter(src,h,'symmetric','same');

end