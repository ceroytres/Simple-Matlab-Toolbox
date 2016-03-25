function deltap=compute_convDelta(delta,W)

sizeDelta=size(delta);
sizeW=size(W);

filterRect=sizeW(1:2);
filterDepth=sizeW(3);
filterNum=sizeW(end);

deltaDim=sizeDelta(1:2);
deltaNum=sizeDelta(3);
numSamples=sizeDelta(end);

%assert(filterNum==deltaNum,'Delta~=numFilters');


deltap=zeros([filterRect+deltaDim-1,filterDepth,numSamples]);

for n=1:numSamples
    for q=1:deltaNum
        deltap(:,:,:,n)=local_conv3(delta(:,:,q,n),W(:,:,:,q),filterDepth)+deltap(:,:,:,n);
    end
end


end


function y=local_conv3(x,h,h_depth)


y=zeros([size(x)+[size(h,1),size(h,2)]-1,h_depth]);

for i=1:h_depth
    y(:,:,i)=conv2(x,h(:,:,i),'full');
end

end



