function [eigvector eigvalue]=my_FastEigen(X,L,r)
% Fast Computation of Standard Eigenvalue Problem X*L*X'
% by QR decomposition
% Input: X -- d x n data matrix
%           L --  Laplacian matrix 
%           r -- reduced dimensionality
% Outpu: eigvector -- eigvectors of  X*L*X' with largest eigvalues
%             eigvalue --- largest eigvalues of X*L*X'
[d,n]=size(X);
if d<n
    disp(['It need not call QR decompostion']);
    S=X*L*X';
    S=(S+S')/2;
    [eigvector,eigvalue]=eig(S);
else
    [Q,R]=qr(X);
    r=find(R(:,end)==0,1)-1;
    Q=Q(:,1:r);
    R=R(1:r,:);
    S=R*L*R';
    S=(S+S')/2;
    [U,eigvalue]=eig(S);
    eigvector=Q*U;
end
if r>d
    error('r > d');
end
eigvalue=diag(eigvalue);
[eigvalue,ind]=sort(eigvalue,'descend');
eigvalue=eigvalue(ind(1:r));
eigvector=eigvector(:,ind(1:r));
