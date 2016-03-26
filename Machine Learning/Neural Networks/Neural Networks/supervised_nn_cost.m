function [ cost, grad,prob] = supervised_nn_cost( theta, ei, data, labels, pred_only)
%
% Cost function for a neural network with L2 weight decay
%


po = false;
if exist('pred_only','var')
  po = pred_only;
end;


stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = stack;
numLayers=numHidden+1;
N=length(labels);


switch ei.activation_fun
    
    case 'logistic'
        f=@(a) 1./(1+exp(-a));
        df=@(z) z.*(1-z);
    case 'tanh'
        f=@(a) tanh(a);
        df=@(z) (1-z.^2);
    case 'funny_tanh'
        f=@(a) 1.7159*tanh((2/3)*a);
        df=@(z) 1.7159*(2/3)*(1-(z/1.7159).^2);
    otherwise
        fprintf('Using default non-linearity\n');
        f=@(a) tanh(a);
        df=@(z) (1-z.^2);
end

% z2=stack{1}.W*data+repmat(stack{1}.b,1,N);
% a2 = f(z2);
% z3 = stack{2}.W*a2 +repmat(stack{2}.b,1,N);
% h=softmax(z3);

% z2=stack{1}.W*data;
% z2=bsxfun(@plus,z2,stack{1}.b);
% a2 = f(z2);
% z3=stack{2}.W*a2;
% z3=bsxfun(@plus,z3,stack{2}.b);
% h=softmax(z3);

%%%%%%%Foward Propagation%%%%%%%%%%%%%%%%%

%%%Input-First Hidden Layer
hAct{1}=stack{1}.W*data;
hAct{1}=f(bsxfun(@plus,hAct{1},stack{1}.b));

%%%Mid-Hidden Layers
for L=2:numLayers-1
    hAct{L}=stack{L}.W*hAct{L-1}; 
    hAct{L}=f(bsxfun(@plus,hAct{L},stack{L}.b)); 
end

%%%Last Hidden-Output layes
hAct{end}=stack{end}.W*hAct{end-1};
hAct{end}=softmax(bsxfun(@plus,hAct{end},stack{end}.b));

prob=hAct{end};

if po
    cost=-1;
    grad=[];
    return;
end



T=sparse(labels,1:N,true,ei.output_dim,N); %%%Convert Labels to 1-K encoding of target values
cost=-sum(log(hAct{end}(T)));  %%%Compute Cross entropy


%%%%%%Backprop%%%%%%

%%%Output/Hidden Layer
delta=hAct{end}-T;
gradStack{end}.W=delta*hAct{end-1}';
gradStack{end}.b=sum(delta,2);

%%%Hidden Layer
for L=numHidden:-1:2
    delta=df(hAct{L}).*(stack{L+1}.W'*delta);
    gradStack{L}.W=delta*hAct{L-1}';
    gradStack{L}.b=sum(delta,2);
end

%%%Input/Hidden Layer
delta=df(hAct{1}).*(stack{2}.W'*delta);
gradStack{1}.W=delta*data';
gradStack{1}.b=sum(delta,2);


%%%% Compute weight decay cost and gradient
for L=1:numLayers
    cost=cost+ei.lambda*sum(stack{L}.W(:).^2);
    gradStack{L}.W=gradStack{L}.W+ei.lambda*stack{L}.W;
end





[grad] = stack2params(gradStack);


end

function f=softmax(a)
a=bsxfun(@minus,a,max(a,[],1));
f=exp(a);
f=bsxfun(@rdivide,f,sum(f,1));
end