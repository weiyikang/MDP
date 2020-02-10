function [W] = myMFA(gnd,k,data,k1,k2)

% 样本，特征数目，类别数目，降维上界
[nSmp,nFea] = size(data);
classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass - 1;

% 计算样本之间的欧氏距离
W_dist = zeros(nSmp,nSmp);
for i = 1:nSmp
    for j = 1:nSmp
      W_dist(i,j) = norm(data(i,:)-data(j,:));
    end
end
      
% 计算本征图权重矩阵W_w与惩罚图权重矩阵W_b
W_w = zeros(nSmp,nSmp);
W_b = zeros(nSmp,nSmp);

for i = 1:nSmp
    dist = W_dist(i,:);
    dist(1,i) = inf;
    [~,idx] = sort(dist);
    count1 = 0;
    count2 = 0;
    for j = 1:nSmp
        if count1 <= k1 && gnd(idx(j))==gnd(i)
            W_w(i,j) = 1;
            count1 = count1 + 1;
        end
        if count2 <= k2 && gnd(idx(j))~=gnd(i)
            W_b(i,j) = 1;
            count2 = count2 + 1;
        end
    end
end

% 计算Laplace矩阵L_w，L_b
L_w = diag(sum(W_w)) - W_w;
L_b = diag(sum(W_b)) - W_b;

Sw = data'*L_w*data;
Sb = data'*L_b*data;

% 分解特征值
A = repmat(0.01,[1,size(Sw,1)]);
B = diag(A);
[V,D] = eig(inv(Sw + B)*Sb);
[~,idx] = sort(diag(D),'descend');
W = [];
for i = 1:k
    W = [W,real(V(:,idx(i)))];
end





