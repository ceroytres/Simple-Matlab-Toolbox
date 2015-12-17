function [u,v]=optical_flow(I1,I2,windowSize,tau)

% [u,v]=optical_flow(I1,I2,windowSize,tau)
%
%  Computes optical flow using the Lucas-Kanade algorithm
%  
%  Input:
%  I1,I2: grayscale images at time t and t+1 respectively
%  windowSize: window size
%  tau: noise threshold
%
%  Output:
%  u,v optical flow fields




h=ones(windowSize);
d=[1/12	-2/3	0	2/3	-1/12];
%d=[-1 0 1]/2;

Ix=imfilter(I2,d,'replicate');
Iy=imfilter(I2,d','replicate');

% Ix=gaussianBlur(Ix,1.5);
% Iy=gaussianBlur(Iy,1.5);

imDim=size(I1);


It=I2-I1;
% It=gaussianBlur(It,1.5);

Ix2=imfilter(Ix.^2,h,'symmetric');
Ixy=imfilter(Ix.*Iy,h,'symmetric');
Iy2=imfilter(Iy.^2,h,'symmetric');

Iyt=imfilter(Iy.*It,h,'symmetric');
Ixt=imfilter(Ix.*It,h,'symmetric');


T=Ix2+Iy2;
D=Ix2.*Iy2-Ixy.^2;
L1 = .5*T + sqrt(.25*T.^2-D);
L2 = .5*T - sqrt(.25*T.^2-D);
L=min(L1,L2);



A=zeros(2,2);
b=zeros(2,1);

u=zeros(imDim);
v=zeros(imDim);

for i=1:imDim(1)
    for j=1:imDim(2)
        if L(i,j)>= tau
            A(1,1)=Ix2(i,j);
            A(1,2)=Ixy(i,j);
            A(2,1)=Ixy(i,j);
            A(2,2)=Iy2(i,j);
              
            b(1,1)=-Ixt(i,j);
            b(2,1)=-Iyt(i,j);
            
            x=A\b;
            u(i,j)=x(1);
            v(i,j)=x(2);
        end
    end
end





end

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