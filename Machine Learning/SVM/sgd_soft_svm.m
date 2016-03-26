function [w]=sgd_soft_svm(x,y,lambda,T)
%  function [w]=sgd_soft_svm(x,y,lambda,T)
%    Fits a soft linear svm usign stochastic subgradient descent
%
%    input:
%         x: input features
%         y: labels 
%         lambda: penalty 
%    Output:
%         w: SVM wieghts


N=size(x,1);
w_t=zeros(size(x,2),1);
w=w_t;

idx_shuffle=randperm(N);
x=x(idx_shuffle,:);
y=y(idx_shuffle);

k=1;
for t=1:T
    
    if y(k)*(x(k,:)*w)<1
        w_t=w_t-1/(t*lambda)*(lambda*w_t-y(k)*x(k,:)');
    else
        w_t=w_t-1/(t*lambda)*(lambda*w_t);
    end
    
    w=(1-1/t)*w+1/t*w_t;
    
    if k==N
        idx_shuffle=randperm(N);
        x=x(idx_shuffle,:);
        y=y(idx_shuffle);
        k=1;
    else
        k=k+1;
    end
    
end


end
