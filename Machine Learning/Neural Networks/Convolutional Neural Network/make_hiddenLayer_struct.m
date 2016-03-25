function hiddenLayer=make_hiddenLayer_struct(act,outDim,inputDim)
hiddenLayer.type='h';

hiddenLayer.outDim=outDim;
hiddenLayer.inputDim=inputDim;

hiddenLayer.wSize=[outDim,inputDim];
hiddenLayer.bSize=outDim;



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



hiddenLayer.f=f;
hiddenLayer.df=df;
hiddenLayer.act_type=act;



end

