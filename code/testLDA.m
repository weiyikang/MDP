% 清除环境变量
clear
clc

% 加载数据
% % 加载Yale数据集
% load('./数据集/Yale_32x32.mat');
% classNum = 15;

% % 加载ORL数据集
% load('./数据集/ORL_32x32.mat');
% classNum = 40;

% 加载YaleB数据集
load('./数据集/YaleB_32x32.mat');
classNum = 38;
ratio = 3;

for i=1:20
    % 划分训练集，测试集
    [X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, classNum, ratio);
    
    % 测试LDA函数
    options = [];
    options.Fisherface = 1;
    [eigvector, eigvalue] = LDA(y_train, options, X_train);
    X_train_lda = X_train*eigvector;
    X_test_lda = X_test*eigvector;
    accuracy(i) = KNN(X_train_lda,y_train,X_test_lda,y_test); 
end

acc = mean(accuracy);
std = std(accuracy);

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


