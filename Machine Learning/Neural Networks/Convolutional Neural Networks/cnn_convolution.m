function features=cnn_convolution(input,layer,W,b)
%Flip kernel
W=W(layer.filterDim(1):-1:1,...
                              layer.filterDim(2):-1:1,...
                              layer.filterDim(3):-1:1,...
                              :);

                          
numFilters=layer.filterDim(end);                          
inputDim=size(input);
convDim=inputDim(1:2)-layer.filterDim(1:2)+1;

%assert(inputDim(3)==layer.filterDim(3),'Depth is not the same!');

features = zeros( [convDim, numFilters, inputDim(4)]);


for i=1:inputDim(4)
    for j=1:numFilters
        features(:,:,j,i)=convn(input(:,:,:,i),W(:,:,:,j),'valid');
        features(:,:,j,i)=layer.f(features(:,:,j,i)+b(j));
    end
end

end