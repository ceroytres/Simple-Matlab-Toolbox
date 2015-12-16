function im=contraharmonicFilter(im,Q,window_size)
%
% function im=contraharmonicFilter(im,Q,window_size)
%       Geometric mean filters image
%       
%       Input:
%            im: input image
%            Q: order 
%            window_size: window size
%
%       Special Cases: Q=0 (mean filter) Q=-1 harmonic mean 
%       Output:
%            im: filtered image

h=ones(window_size);
im=imfilter(im.^(Q+1),h)./imfilter(im.^Q,h);

end