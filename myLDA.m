function [W] = myLDA(gnd,k,data)

% ������������Ŀ�������Ŀ����ά�Ͻ�
[nSmp,nFea] = size(data);
classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass - 1;

% Sw,Sb
Sw = zeros(nFea,nFea);
Sb = zeros(nFea,nFea);

% �����ܾ�ֵ
sampleMean = mean(data,1);

for i = 1:nClass
    idx = gnd==i;
    per_class = data(idx,:);
    per_mean = mean(per_class,1);
    % ����������ɢ��
    temp = per_class - repmat(per_mean,size(per_class,1),1);
    Sw = Sw + temp'*temp;
    % ���������ɢ��
    Sb = Sb + size(per_class,1)*(per_mean-sampleMean)'*(per_mean-sampleMean);
end

Sw = Sw/nSmp;
Sb = Sb/nSmp;

% �ֽ�����ֵ
A = repmat(0.001,[1,size(Sw,1)]);
B = diag(A);
[V,D] = eig(inv(Sw+B)*Sb);
[~,idx] = sort(diag(D),'descend');
W = [];
for i = 1:k
    W = [W,V(:,idx(i))];
end




