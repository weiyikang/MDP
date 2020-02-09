function [W] = myMDP(gnd,k,data,k1,k2)

% ������������Ŀ�������Ŀ����ά�Ͻ�
[nSmp,nFea] = size(data);
classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass - 1;

% Sw,Sb
Sw = zeros(nFea,nFea);
Sb = zeros(nFea,nFea);

% �������ڣ������ɢ��Sw,Sb
for i = 1:nClass
    idx = gnd==i;
    per_class = data(idx,:);
    per_mean = mean(per_class,1);
    
    % ����������ɢ��
    % ����������ֵ��ͬ��������ŷ�Ͼ���
    dist = [];
    for j = 1:nSmp
        dist = [dist,norm(data(j)-per_mean)];
    end
    
    % ����Sw,Sb
    [~,idx1] = sort(dist,'descend');
    [~,idx2] = sort(dist);
    count1 = 0;
    count2 = 0;
    for j = 1:nSmp
        if count1 > k1 && count2 > k2
            break;
        end
        if count1 <= k1 && gnd(idx1(j))==i
            Sw = Sw + (data(gnd(idx1(j)))-per_mean)'*(data(gnd(idx1(j)))-per_mean);
            count1 = count1 + 1;
        end
        if count2 <= k2 && gnd(idx2(j))==i
            Sb = Sb + (data(gnd(idx2(j)))-per_mean)'*(data(gnd(idx2(j)))-per_mean);
            count2 = count2 + 1;
        end
    end
end

Sw = Sw/(nClass*k1);
Sb = Sb/(nClass*k2);

% �ֽ�����ֵ
[V,D] = eig(Sb-Sw);
% A = repmat(0.01,[1,size(Sw,1)]);
% B = diag(A);
% [V,D] = eig(inv(Sw+B)*Sb);
[~,idx] = sort(diag(D),'descend');
W = [];
for i = 1:k
    W = [W,real(V(:,idx(i)))];
end


