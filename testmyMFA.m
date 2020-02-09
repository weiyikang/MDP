% �����������
clear
clc

% ��������
load('./���ݼ�/Yale_32x32.mat');
classNum = 15;
% ����ѵ���������Լ�
[X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, classNum, 0.3);

% �Ƚ���PCAԤ�������������������
% options=[];
% options.ReducedDim=70;
% [eigvector,eigvalue] = PCA(X_train,options);
% X_train = X_train*eigvector;
% X_test = X_test*eigvector;

% ����MFA
k = 16;
k1 = 2;
k2 = 1;
[W] = myMFA(y_train, k, X_train, k1, k2);
X_train_mfa = X_train*W;
X_test_mfa = X_test*W;

% ʹ��knnԤ����
mdl = fitcknn(X_train_mfa, y_train, 'NumNeighbors', 1);
total_num = size(X_test_mfa, 1);
acc = 0;
for i = 1:total_num
    label = predict(mdl, X_test_mfa(i,:));
    if label == y_test(i);
        acc = acc + 1;
    end
end

acc = acc/total_num

% % ��ʾͼƬ
% faceW = 32; 
% faceH = 32; 
% numPerLine = 10; 
% ShowLine = 10; 
% 
% Y = zeros(faceH*ShowLine,faceW*numPerLine); 
% for i=0:ShowLine-1 
%   	for j=0:numPerLine-1 
%     	Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(fea(i*numPerLine+j+1,:),[faceH,faceW]); 
%   	end 
% end 

% imagesc(Y);colormap(gray);



