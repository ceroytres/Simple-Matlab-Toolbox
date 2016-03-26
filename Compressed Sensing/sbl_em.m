function [x,S]=sbl_em(A,b)
%% Sparse Bayesian Learning EM algorithm
%%
%%
maxIter=1e4;
c=size(A,2);
g=abs(pinv(A)*b)+1;
I=eye(c);


xp=g;
r=inf;
j=1;
while r>1e-9
    
    %E Step:
    g_root=(sqrt(g));
    B=pinv(bsxfun(@times,A,g_root'));
    x=g_root.*(B*b);
    
    S=(I-bsxfun(@times,g_root,B*A));
    S=bsxfun(@times,S,g');
    S=.5*(S+S);
    
    %M Step:
    g=x.^2+abs(diag(S));
    
    %Compute stopping condition
    r=norm(x-xp,2);
    xp=x;
    
    if(j>maxIter)
        break;
    else
        j=j+1;
    end
end


g_root=(sqrt(g));
B=pinv(bsxfun(@times,A,g_root'));
x=g_root.*(B*b);

x(abs(x)<1e-6)=0;
S=find(x);
end