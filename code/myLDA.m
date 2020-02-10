function [W] = myLDA(gnd,k,data)

% 样本，特征数目，类别数目，降维上界
[nSmp,nFea] = size(data);
classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass - 1;

% Sw,Sb
Sw = zeros(nFea,nFea);
Sb = zeros(nFea,nFea);

% 样本总均值
sampleMean = mean(data,1);

for i = 1:nClass
    idx = gnd==i;
    per_class = data(idx,:);
    per_mean = mean(per_class,1);
    % 计算类内离散度
    temp = per_class - repmat(per_mean,size(per_class,1),1);
    Sw = Sw + temp'*temp;
    % 计算类间离散度
    Sb = Sb + size(per_class,1)*(per_mean-sampleMean)'*(per_mean-sampleMean);
end

Sw = Sw/nSmp;
Sb = Sb/nSmp;

% 分解特征值
A = repmat(0.001,[1,size(Sw,1)]);
B = diag(A);
[V,D] = eig(inv(Sw+B)*Sb);
[~,idx] = sort(diag(D),'descend');
W = [];
for i = 1:k
    W = [W,V(:,idx(i))];
end




