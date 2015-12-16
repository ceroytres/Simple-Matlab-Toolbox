function im=geometricFilter(im,window_size)
%
% function im=geometricFilter(im,window_size)
%       Geometric mean filters image
%       
%       Input:
%            im: input image
%            window_size: window size
%       Output:
%              im:filtered image


MN=prod(window_size);
im(im<eps)=eps;
im=log(im); % Convert into log domain

h=ones(window_size)/MN;
im=imfilter(im,h); % mean filter
im=exp(im); % convert into image domain

end