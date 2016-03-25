function outLayer=make_outLayer_struct(numClasses,inputDim)
outLayer.type='o';
outLayer.f=@(a) softmax(a);
outLayer.outDim=numClasses;
outLayer.inputDim=inputDim;

outLayer.wSize=[numClasses,inputDim];
outLayer.bSize=numClasses;
outLayer.act_type='softmax';
end

function z=softmax(a)
a=bsxfun(@minus,a,max(a,[],1));
z=exp(a);
z=bsxfun(@rdivide,z,sum(z,1));
end