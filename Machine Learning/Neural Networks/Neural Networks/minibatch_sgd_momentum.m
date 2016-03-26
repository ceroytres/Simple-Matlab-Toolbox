function [theta,f]=minibatch_sgd_momentum(fObj,theta,X,y,options)

if nargin==4
    options=[];
end


T=getOpts(options,'T',100);
eta=getOpts(options,'eta',1e-3);
batchSize=getOpts(options,'batchSize',10);
method=getOpts(options,'method','nag');
tol=getOpts(options,'tol',1e-5);

N=size(X,2);
numBatches=ceil(N/batchSize);


v=0;
t=0;


loop_flag=false;
iter=2;
f=fObj(theta,X,y);
for i=1:T
    batches=make_batches(N,batchSize);
    
    for b=1:numBatches
        theta_old=theta;
        
        mu=1-(3/(t+5));
        
        if strcmpi(method,'nag')
            [~,g,~]=fObj(theta+mu*v,X(:,batches{b}),y(batches{b}));
        elseif strcmpi(method,'off')
            [~,g,~]=fObj(theta,X(:,batches{b}),y(batches{b}));
            mu=0;
        else
            [~,g,~]=fObj(theta,X(:,batches{b}),y(batches{b}));
        end
        
        v=mu*v-eta*g;
        
        theta=theta+v;
        t=t+1;
        
       
        f(iter)=fObj(theta,X,y);
        iter=iter+1;
        
        
        progTol=sqrt(sum((theta-theta_old).^2));
        
        
        if progTol<tol
            fprintf('progTol %f\n',progTol);
            loop_flag=true;
            break;
        end
        
    end
      
    if loop_flag
       break; 
    end
    
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