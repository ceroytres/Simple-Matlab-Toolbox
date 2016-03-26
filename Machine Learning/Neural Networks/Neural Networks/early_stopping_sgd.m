function [best_params,best_epoch,best_validation_loss]=early_stopping_sgd(fObj,theta,train,validation,options)
%
% Implemenets an early stopping heuristic based on "patience"
%
% Input
%     fObj: Objective function handle which returns objective function
%           value,gradient, and prediction values!
%     theta: initial guess
%     train: train.X should include training set 
%            train.y should include training labels
%     validation:
%            validation.X should include vaildation set 
%            validation.y should include vaildation labels set
%     options:
%          struct containing
%               batchSize(default=256)
%               patience(default=5000)
%               patience_increase(default=2)
%               improvement_threshold(default=.995)
%               validation_freq(default=min(numBatches,patience/2)))
%               T(default=10)
%               eta(default=1e-3)
%       
%     Borrowed from the Theano 

%%%%% Setup Early-Stopping Parameters
N=size(train.X,2);

batchSize=getOpts(options,'batchSize',10);
numBatches=ceil(N/batchSize);


patience=getOpts(options,'patience',5000);
patience_increase=getOpts(options,'patience_increase',2);
improvement_threshold=getOpts(options,'improvement_threshold',.995);
validation_freq=getOpts(options,'validation_freq',...
    min(numBatches,patience/2));
T=getOpts(options,'T',10);
eta=getOpts(options,'eta',1e-3);

best_params=[];
best_epoch=[];
best_validation_loss=inf;

epoch=0;

loop_flag=true;
iter=0;
v=0;
while (epoch<T) & (loop_flag)
    epoch=epoch+1;
    batches=make_batches(N,batchSize);
    
    for b=1:numBatches
        mu=1-(3/(iter+5));
        [~,g,~]=fObj(theta,train.X(:,batches{b}),train.y(batches{b}));
        
        v=v*mu-eta*g;
        
        theta=theta+v;
        iter=(epoch-1)*numBatches+b;
        
        if mod(iter+1,validation_freq)==0
            [~,~,prob]=fObj(theta,validation.X,validation.y);
            [~,pred]=max(prob,[],1);
            loss=mean(pred(:)~=validation.y(:));
            
            
            %fprintf('loss:%d\n',loss);
            if loss<best_validation_loss
                if loss<(best_validation_loss*improvement_threshold)
                    patience=max(patience,iter*patience_increase);
                end
                best_epoch=epoch;
                best_validation_loss=loss;
                best_params=theta;
            end
        end
        
        if patience <= iter
            loop_flag=false;
            break;
        end
        
        
    end
    

end

end

%%%%%%%%%      auxiliary functions   %%%%%%%%%%%%%%%%%%%%%%

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