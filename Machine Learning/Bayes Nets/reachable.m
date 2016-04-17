function R=reachable(G,X,Z)
%
% R=reachable(G,X,Z)
% Finds all reachable nodes which are not d-seperated from X given Z
%  G: Bayesian Network Graph bulit using dipgraph in MATLAB
%  X: Source node
%  Z: Set of evidence nodes
%
%  R:Set of reachable nodes
%
% Example:
% A=[0 0 0 0 1;
%    0 0 0 1 0;
%    0 0 0 1 0;
%    0 0 0 0 1;
%    0 0 0 0 0];
% G=digraph(A,{'M','S','R','P','A'});
% plot(G)
% R=reachable(G,'M',py.set(['A','P']));
% disp(R)
% 
% 
%
%Based on:
%Probabilistic Graphical Models – Principles and Techniques
%Algorithm 3.1 – Algorithm for finding nodes reachable from X given Z via active trails
%Koller, Nir
%
L=Z.copy();
A=py.set();


while py.len(L)~=0
    Y=char(L.pop());
    
    if py.len(A.intersection(py.set(Y)))==0
        
        paY=G.predecessors(char(Y))';
        
        if ~isempty(paY)
            L=L.union(py.tuple(paY));
        end
    end
    
    A=A.union(Y);
end

L=py.set(py.list({py.tuple({X,'u'})}));
V=py.set();
R=py.set();

while py.len(L) ~= 0
    Y=L.pop();
    
    if py.len(V.intersection(py.set({py.tuple(Y)})))==0
        if py.len(Z.intersection(py.set(Y{1})))==0
            R=R.union(Y{1});
        end
        V=V.union({Y});
        
        if Y{2}=='u' && py.len(Z.intersection(py.set(Y{1})))==0
            
            paY=G.predecessors(char(Y{1}));
            
            for i=1:length(paY)
                L=L.union(py.list({py.tuple({paY{i},'u'})}));
            end
            
            ChY=G.successors(char(Y{1}));
            
            for i=1:length(ChY)
                L=L.union(py.list({py.tuple({ChY{i},'d'})}));
            end
            
        elseif  Y{2}=='d'
            if py.len(Z.intersection(py.set(Y{1})))==0
                ChY=G.successors(char(Y{1}));
                
                for i=1:length(ChY)
                    L=L.union(py.list({py.tuple({ChY{i},'d'})}));
                end
            end
            
            if py.len(A.intersection(py.set(Y{1})))>0
                paY=G.predecessors(char(Y{1}));
                
                for i=1:length(paY)
                    L=L.union(py.list({py.tuple({paY{i},'u'})}));
                end
            end
            
        end
    end
    
end




end
