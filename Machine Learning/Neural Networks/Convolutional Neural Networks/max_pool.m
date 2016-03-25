function [pooledFeatures,mask]=max_pool(poolDim,features)


[r,c,numFilters,numImages] = size(features);

if any(floor([r,c]./ poolDim)~=([r,c]./ poolDim))
    error('Input size not divisible pool window! Catz rule');
end
poolSize=([r,c]./ poolDim);

pooledFeatures = zeros([poolSize, numFilters, numImages]);

mask=false(size(features));




%[x_lut,y_lut]=ind2sub(poolDim,1:prod(poolDim));


for i=1:numImages
    for j=1:numFilters
        
        a=1; b=poolDim(1); c=1; d=poolDim(2);
        
        for k=1:poolSize(1)
            
            for t=1:poolSize(2)
                
                pooledFeatures(k,t,j,i)=max(max(features(a:b,c:d,j,i)));
                [x,y]=find(features(a:b,c:d,j,i)==pooledFeatures(k,t,j,i));
                
%                 idx=(features(a:b,c:d,j,i)==pooledFeatures(k,t,j,i));
%                 
%                 idx1=x_lut(idx)+(k-1)*poolDim(1);
%                 idx2=y_lut(idx)+(t-1)*poolDim(2);
                
                idx1=x(1)+(k-1)*poolDim(1);
                idx2=y(1)+(t-1)*poolDim(2);
                
                mask(idx1,idx2,j,i)=true;
                
                c=c+poolDim(2);
                d=d+poolDim(2);
            end
            c=1;
            d=1;
            a=a+poolDim(1);
            b=b+poolDim(1);
        end
        
        
        
    end
end





end