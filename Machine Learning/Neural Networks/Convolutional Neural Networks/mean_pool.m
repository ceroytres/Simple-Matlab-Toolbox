function pooledFeatures = mean_pool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

[r,c,numFilters,numImages] = size(convolvedFeatures);



pooledFeatures = zeros([floor([r,c]./ poolDim), numFilters, numImages]);

h=ones(poolDim)/prod(poolDim);

for i=1:numImages
    for j=1:numFilters
    temp=conv2(convolvedFeatures(:,:,j,i),...
        h,'valid');
    pooledFeatures(:,:,j,i)=temp(1:poolDim(1):end,1:poolDim(2):end);
    end
end


end