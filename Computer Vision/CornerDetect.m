function corners = CornerDetect(I, nCorners,R, smoothSTD, windowSize)
% corners = CornerDetect(I, nCorners,R, smoothSTD, windowSize)
%
% Computes corner locations using the Shi-Tomasi Corner Detector
%
% Input:
%   I: grayscale image
%   nCorners: number of desired corners
%   R: radius of non-max supression
%   smoothSTD: sigma for Gaussian smoothing
%   windowSize: window size
% Output:
%   corners: corner locations

d=[1/12	-2/3	0	2/3	-1/12];

I=gaussianBlur(I,smoothSTD);

Ix=imfilter(I,d,'replicate');
Iy=imfilter(I,d','replicate');


h=ones(windowSize);
Ix2=imfilter(Ix.^2,h,'symmetric');
Ixy=imfilter(Ix.*Iy,h,'symmetric');
Iy2=imfilter(Iy.*Iy,h,'symmetric');

T=Ix2+Iy2;
D=Ix2.*Iy2-Ixy.^2;
lambda1 = .5*T + sqrt(.25*T.^2-D);
lambda2 = .5*T - sqrt(.25*T.^2-D);
lambda_min=min(lambda1,lambda2);


cornerMap=nonMax_suppression(lambda_min,R);

corners=find_best_corners(cornerMap,nCorners);
end


function corner_map=nonMax_suppression(corner_map,R)


seSize=2*R+1;
se=ones([seSize,seSize]);


corner_dilated=imdilate(corner_map,se);
corner_map(corner_map<corner_dilated)=0;

candidate_map=(corner_map>0);
candidate_map=bwmorph(candidate_map,'shrink',inf);
corner_map=corner_map.*candidate_map;

end

function loc=find_best_corners(corner_map,nCorners)
mapSize=size(corner_map);
[~,candidates]=sort(corner_map(:),'descend');
[i,j]=ind2sub(mapSize,candidates(1:nCorners));
loc=[j,i];
end