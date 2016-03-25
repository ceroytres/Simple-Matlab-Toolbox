function [cost,grad,predictions]=cnn_cost(x,t,layers,theta,pred)

batchSize=size(x,4);
if ~exist('pred','var')
    pred = false;
end;



if exist('theta','var')
    layers{end}.theta=theta;
end



%%% Allocate resources
numLayers=length(layers)-1;
stack=params2stack(layers);
stack_grad=stack;
act=cell(numLayers,1);
poolAct=cell(layers{end}.numConv,1);

if layers{end}.numMax>0
    poolMask=cell(layers{end}.numMax);
    maxCounter=0;
end

first_Hidden_layer=true;


poolCounter=0;

for i=1:numLayers
   switch lower(layers{i}.type)
       case 'c'
           if(i==1)
               act{1}=cnn_convolution(x,layers{1},stack{1}.W,stack{1}.b);
           else
               act{i}=cnn_convolution(act{i-1},layers{i},stack{i}.W,stack{i}.b); 
           end
           
           switch lower(layers{i}.pool_type)
               case 'mean'
                   poolCounter=poolCounter+1;
                   poolAct{poolCounter}=mean_pool(layers{i}.pool_dim,act{i});
               case 'max'
                   poolCounter=poolCounter+1;
                   maxCounter=maxCounter+1;
                   [poolAct{poolCounter},poolMask{maxCounter}]=max_pool(layers{i}.pool_dim,act{i});
               case 'none'
                   poolCounter=poolCounter+1;
                   poolAct{poolCounter}=act{i};
               otherwise
                   error('Invalid pooling');
           end
           
       case 'h'
           
           if(first_Hidden_layer)
                first_Hidden_layer=~first_Hidden_layer;
                act{i}=stack{i}.W*reshape(poolAct{end},[],batchSize);
                act{i}=layers{i}.f(bsxfun(@plus,act{i},stack{i}.b));
           else
                act{i}=stack{i}.W*act{i-1};
                act{i}=layers{i}.f(bsxfun(@plus,act{i},stack{i}.b));
           end
           
       case 'o'
           
           switch lower(layers{i-1}.type)
               case 'c'
                   act{i}=stack{i}.W*reshape(poolAct{end},[],batchSize);
                   act{i}=layers{i}.f(bsxfun(@plus,act{i},stack{i}.b));
               case 'h'
                   act{i}=stack{i}.W*act{i-1};
                   act{i}=layers{i}.f(bsxfun(@plus,act{i},stack{i}.b));
               otherwise
                   error('Katz!');     
           end
           
       otherwise
           error('invaild layer');
   end
       
end





if(pred)
    grad=[];
    cost=[];
    [~,predictions]=max(act{end},[],1);
    return;
else
    predictions=[];
end

T=sparse(t,1:batchSize,true,layers{end-1}.outDim,batchSize); 
cost=-mean(log(act{end}(T)));


for i=numLayers:-1:1
    switch lower(layers{i}.type)
        case 'o'
            delta=(act{end}-T)/batchSize;
            stack_grad{i}.b(:)=sum(delta,2);
            
            switch lower(layers{i-1}.type)
                case 'c'
                    stack_grad{i}.W=delta*reshape(poolAct{end},[],batchSize)';
                case 'h'
                    stack_grad{i}.W=delta*act{i-1}';
                otherwise
                    error('Invalid config');
            end
            delta=stack{i}.W'*delta;
        case 'h'
            delta=delta.*layers{i}.df(act{i});
            stack_grad{i}.b(:)=sum(delta,2);
            
            switch lower(layers{i-1}.type)
                case 'c'
                    stack_grad{i}.W=delta*reshape(poolAct{end},[],batchSize)';
                case 'h'
                    stack_grad{i}.W=delta*act{i-1}';
                otherwise
                    error('Invalid config');
            end
            
            delta=stack{i}.W'*delta;
        case 'c'
            delta=reshape(delta,[layers{i}.outDim,batchSize]);
            
            switch lower(layers{i}.pool_type)
                case 'mean'
                    delta=mean_poolUpsample(delta,layers{i}.pool_dim);
                case 'max'
                    delta=max_poolUpsample(delta,poolMask{maxCounter},layers{i}.pool_dim);
                    maxCounter=maxCounter-1;
                otherwise
                    if ~strcmpi(layers{i}.pool_type,'none')
                        error('Error invalid pooling layer');
                    end
            end
            
            delta=delta.*layers{i}.df(act{i});
            
            stack_grad{i}.W=stack_grad{i}.W*0;
            stack_grad{i}.b=stack_grad{i}.b*0;
            
            if i>1
                poolCounter=poolCounter-1;
                for t=1:layers{i}.filterDim(end)
                    for k=1:batchSize
                        h=rot90(delta(:,:,t,k),2);
                        stack_grad{i}.W(:,:,:,t)=convn(poolAct{poolCounter}...
                            (:,:,:,k),h,'valid')+stack_grad{i}.W(:,:,:,t);
                        stack_grad{i}.b(t)=sum(h(:))+stack_grad{i}.b(t);
                    end
                end
                
                delta=compute_convDelta(delta,stack{i}.W);
              
              else
                for t=1:layers{i}.filterDim(end)
                    for k=1:batchSize
                        h=rot90(delta(:,:,t,k),2);
                        stack_grad{i}.W(:,:,:,t)=convn(x(:,:,:,k)...
                            ,h,'valid')+stack_grad{i}.W(:,:,:,t);
                        stack_grad{i}.b(t)=sum(h(:))+stack_grad{i}.b(t);
                    end
                end
            end
           
        otherwise
            error('Layer invalid');
    end
end


grad=stack2params(stack_grad);

end


function delta_pool=mean_poolUpsample(delta_w,poolDim)
N=prod(poolDim);
poolMask=ones(poolDim)/N;

sizeDelta=size(delta_w);

delta_pool=zeros([sizeDelta(1:2).*poolDim,sizeDelta(3),sizeDelta(4)]);


for i=1:sizeDelta(end)
    for j=1:sizeDelta(end-1)
        delta_pool(:,:,j,i)=kron(delta_w(:,:,j,i),poolMask);
    end
end


end


function delta_pool=max_poolUpsample(delta_w,mask,poolDim)

poolMask=ones(poolDim);
sizeDelta=size(delta_w);

delta_pool=zeros([sizeDelta(1:2).*poolDim,sizeDelta(3),sizeDelta(4)]);


for i=1:sizeDelta(end)
    for j=1:sizeDelta(end-1)
        delta_pool(:,:,j,i)=kron(delta_w(:,:,j,i),poolMask);
    end
end

delta_pool=delta_pool.*double(mask);
end

