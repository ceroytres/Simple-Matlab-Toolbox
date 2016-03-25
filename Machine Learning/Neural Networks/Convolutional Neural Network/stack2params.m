function theta=stack2params(stack)

paramsTotal=stack{end};



theta=zeros(paramsTotal,1);

idx=1;

numLayers=length(stack)-1;

for i=1:numLayers
    numOfparams=numel(stack{i}.W);
    
    theta(idx:idx+numOfparams-1)=stack{i}.W(:);
    
    idx=idx+numOfparams;
    
    numOfparams=numel(stack{i}.b);
    
    theta(idx:idx+numOfparams-1)=stack{i}.b(:);
    
    idx=idx+numOfparams;
end

end