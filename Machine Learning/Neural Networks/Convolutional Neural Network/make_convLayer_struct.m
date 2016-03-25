function convLayer=make_convLayer_struct(act,inputDim,filterSize,pool_type,pool_dim)
% function convLayer...
%          make_convLayer_struct(act,filterSize,filterNum,pool_type,pool_dim)
% act: activation type
% inputDim:[r,c,depth]
% filterSize: [r,c,depth,numFilter]
% pool_type: max or mean none


convLayer.type='c';



convLayer.filterDim=filterSize;
convLayer.pool_type=pool_type;
convLayer.pool_dim=pool_dim;


convLayer.inputDim=inputDim;


if ~strcmpi(pool_type,'none')
    convLayer.outDim=[floor((inputDim(1:2)-convLayer.filterDim(1:2)+1)./pool_dim),...
        filterSize(end)];
else
    convLayer.pool_dim=[];
        convLayer.outDim=[(inputDim(1:2)-convLayer.filterDim(1:2)+1),...
        filterSize(end)];
end

convLayer.wSize=filterSize;
convLayer.bSize=filterSize(end);



switch lower(act)
    case 'logistic'
        f=@(a) 1./(1+exp(-a));
        df=@(z) z.*(1-z);

    case 'tanh'
        f=@(a) tanh(a);
        df=@(z) 1-z.^2;

    case 'relu'
        f=@(a) max(0,a);
        df=@(z) double(z>0);

    case 'soft_plus'
        f=@(a) log(1+exp(a));
        df=@(z) 1+exp(-z);
        
    case 'funny_tanh'
        f=@(a) 1.7159*tanh((2/3)*a);
        df=@(z) 1.7159*(2/3)*(1-(z/1.7159).^2);
        
    otherwise
        warning('Selected invalid activation unit using ReLu!!!');
        f=@(a) max(0,a);
        df=@(z) double(z>0);
        
end

convLayer.f=f;
convLayer.df=df;
convLayer.act_type=act;
end

