function [eigvector, eigvalue]=my_MDP(X,Label,r)
% Idea: Maximizing mimimum interclass distance
%          & Mimizing maximum intraclass distance
% Input:    X -- data matrix d x n
%              Label -- n x1 label
%              r -- reduced dimensionality
% Output:  
%               eigvector -- d x r eigen vectors for projection
%               eigvalue -- r eigvalues 
[d n]=size(X);
if nargin==2
    r=d;
end
% computing pairwise similaity
D=EuDist2(X');

% computing affinity matrix
W1 = zeros(n, n);   % intra-class affinity matrix
W2 = zeros(n, n);   % inter-class affinity matrix
% computing maximum intra-class distance
for i = 1 : n
    id=find(Label==Label(i));
    temp=D(i,id');
    [~, ind]=max(temp);
    W1(i, id(ind(1))') = 1; 
end
W1=max(W1,W1');
% computing minimum inter-class distance
for i = 1 : n
    id=find(Label~=Label(i));
    temp=D(i,id');
    [~, ind]=min(temp);
    W2(i, id(ind(1))') = 1; 
end
W2=max(W2,W2');

% intra-class Laplacian matrix L_intra
% inter-class Laplacian matrix L_inter
temp1=sum(W1,2);
temp2=sum(W2,2);
L_intra=-1*W1;
L_inter=-1*W2;
for i=1:length(temp1)
    L_intra(i,i)=L_intra(i,i)+temp1(i);
    L_inter(i,i)=L_inter(i,i)+temp2(i);
end

L_inter = 0.5*(L_inter + L_inter');
L_intra = 0.5*(L_intra + L_intra');
regu=0.5;
L=regu*L_inter-(1-regu)*L_intra;

[eigvector,eigvalue]=my_FastEigen(X,L,r);

% options=[];
% options.PCARatio = 1;
% options.ReducedDim=r;
% [eigvector, eigvalue] = LGE(L_inter,[], options, X');