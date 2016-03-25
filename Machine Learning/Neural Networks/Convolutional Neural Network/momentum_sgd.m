function [theta,J]=momentum_sgd(fObj,theta,x,y,options)

epochs=getOpts(options,'epochs',3);
alpha=getOpts(options,'alpha',1e-1);
mu0=getOpts(options,'mu0',1);
minibatch=getOpts(options,'minibatch',256);
annel=getOpts(options,'annel',2);

N=length(y);
numBatches=ceil(N/minibatch);
v=zeros(size(theta));
J=zeros(1,numBatches);


t=0;

for T=1:epochs
    batches=make_batches(N,minibatch);
    
    for b=1:numBatches
        t=t+1;
        
        x_batch=x(:,:,:,batches{b});
        y_batch=y(batches{b});
        
            [J(t) g]=fObj(theta,x_batch,y_batch);
        
        mu=mu0*(1-3/(t-1+5));
        
        v=-alpha*g+mu*v;
        theta=theta+v;
        
        
        if ~mod(t,floor(minibatch/10));
            fprintf('Epoch: %d iteration: %d Cost: %f\n',T,t,J(t));
        end
        
        
    end
    
    alpha=alpha/annel;
end

end

function v=getOpts(options,opt,default)
%%%Function borrowed from ptmk 3 toolbox by Kevin Murphy

if isfield(options,opt)
    if ~isempty(getfield(options,opt))
        v = getfield(options,opt);
    else
        v = default;
    end
else
    v = default;
end

end


function batches=make_batches(N,batchSize)

numBatches=ceil(N/batchSize);

batches=cell(numBatches,1);

idx=randperm(N);

k=N;
for i=1:numBatches
    if k-batchSize>0
        start_idx=((i-1)*batchSize)+1;
        end_idx=start_idx+batchSize-1;
    else
        start_idx=end_idx+1;
        end_idx=N;
    end
    
    
    batches{i}=idx(start_idx:end_idx);
    k=k-batchSize;
end

end
