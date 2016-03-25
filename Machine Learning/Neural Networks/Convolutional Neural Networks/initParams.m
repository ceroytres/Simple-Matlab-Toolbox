function layers=initParams(layers)



numLayers=length(layers)-1;

totalParams=0;
for i=1:numLayers
    totalParams=prod(layers{i}.wSize)+totalParams+layers{i}.bSize; 
end

layers{end}.theta=zeros(totalParams,1);


idx_start=1;
layers{end}.numConv=0;
layers{end}.numMax=0;
for i=1:numLayers
    
    
    if strcmpi(layers{i}.type,'c')
        totalParams=prod(layers{i}.wSize);
        layers{end}.theta(idx_start:idx_start+totalParams-1)=...
            .1*randn(totalParams,1);
        layers{end}.numConv=layers{end}.numConv+1;
        
        if strcmpi(layers{i}.pool_type,'max')
           layers{end}.numMax=layers{end}.numMax+1;
        end

        
    else
        totalParams=prod(layers{i}.wSize);
        r=sqrt(6)/sqrt(prod(layers{i}.inputDim)+prod(layers{i}.outDim)+1);
        layers{end}.theta(idx_start:idx_start+totalParams-1)=...
            (rand(totalParams,1)-.5)*2*r;
    end
    
    
    idx_start=idx_start+totalParams;
    
    if  strcmpi(layers{i}.act_type,'relu')
        totalParams=layers{i}.bSize;
        layers{end}.theta(idx_start:idx_start+totalParams-1)=...
            ones(totalParams,1);
        
    else
        totalParams=layers{i}.bSize;
        layers{end}.theta(idx_start:idx_start+totalParams-1)=...
            zeros(totalParams,1);
        
    end
    
    idx_start=idx_start+totalParams;
end


end