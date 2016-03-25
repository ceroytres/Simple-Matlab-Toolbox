function stack=params2stack(layers)


numLayers=length(layers);


stack=cell(numLayers,1);


idx=1;
for i=1:numLayers-1
    t=prod(layers{i}.wSize);
    stack{i}.W=reshape(layers{end}.theta(idx:idx+t-1),layers{i}.wSize);
    idx=idx+t;
    
    stack{i}.b=layers{end}.theta(idx:idx+layers{i}.bSize-1);
    idx=idx+layers{i}.bSize;
end

totalParams=0;
for i=1:numLayers-1
    totalParams=prod(layers{i}.wSize)+totalParams+layers{i}.bSize; 
end


stack{end}=totalParams;


end