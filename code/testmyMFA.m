% 清除环境变量
clear
clc

% 加载数据
load('./数据集/Yale_32x32.mat');
classNum = 15;
% 划分训练集，测试集
[X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, classNum, 0.3);

% 先进行PCA预处理，避免矩阵奇异问题
% options=[];
% options.ReducedDim=70;
% [eigvector,eigvalue] = PCA(X_train,options);
% X_train = X_train*eigvector;
% X_test = X_test*eigvector;

% 测试MFA
k = 16;
k1 = 2;
k2 = 1;
[W] = myMFA(y_train, k, X_train, k1, k2);
X_train_mfa = X_train*W;
X_test_mfa = X_test*W;

% 使用knn预测结果
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

% % 显示图片
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



