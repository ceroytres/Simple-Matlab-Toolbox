function [x,S]=mp(A,b,tol)
%
% Solves min ||x||_0 s.t. Ax=b
% using matching pursuit
% input:
%       A: sensing matrix
%       b: observation vector
%       tol: tolerance on residual norm



W=sqrt(sum(A.*conj(A),1)); 
A=bsxfun(@rdivide,A,W);

x=zeros(size(A,2),1);
S=[];


r=b;
normr=sqrt(r'*r);


while normr>tol
    
    z=conj(r'*A);

    [~,k]=max(abs(z));
    
    S=[S,k];
    
    r=r-z(k)*A(:,k);

    x(k)=x(k)+z(k);

    normr=norm(r);
    
end

S=unique(S);
x=x./W';

end