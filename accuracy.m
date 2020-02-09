% �����������
clear
clc

% ��������
load('./���ݼ�/Yale_32x32.mat');

% ����ѵ���������Լ�
[X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, 15, 0.3);

% % ����PCA����
% options=[];
% options.ReducedDim=50;
% [eigvector,eigvalue] = PCA(X_train,options);
% X_train_pca = X_train*eigvector;
% X_test_pca = X_test*eigvector;

% ����LDA����
options = [];
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(y_train, options, X_train);
X_train_lda = X_train*eigvector;
X_test_lda = X_test*eigvector;

% ʹ��knnԤ����
mdl = fitcknn(X_train_lda, y_train, 'NumNeighbors', 1);
total_num = size(X_test_lda, 1);
acc = 0;
for i = 1:total_num
    label = predict(mdl, X_test_lda(i,:));
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
% 
% imagesc(Y);colormap(gray);



