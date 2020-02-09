% �����������
clear all
clc

% ��������
% ����Yale���ݼ�
% load('./���ݼ�/Yale_32x32.mat');
% classNum = 15;

% ����ORL���ݼ�
load('./���ݼ�/ORL_32x32.mat');
classNum = 40;

% % ����YaleB���ݼ�
% load('./���ݼ�/YaleB_32x32.mat');
% classNum = 38;
ratio = 5;

for dim=1:45
    for i=1:10
        % ����ѵ���������Լ�
        [X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, classNum, ratio);
        
        % ����PCA����
        options=[];
        options.ReducedDim=dim;
        [eigvector,eigvalue] = PCA(X_train,options);
        X_train_pca = X_train*eigvector;
        X_test_pca = X_test*eigvector;
        accuracy(i) = KNN(X_train_pca,y_train,X_test_pca,y_test,1);
    end
    acc(dim) = mean(accuracy);
    std_acc(dim) = std(accuracy);
end

% acc = mean(accuracy);
% std = std(accuracy);
path = ['Yale_L',num2str(ratio),'_acc_1to45_pca'];
path_orl = ['ORL_L',num2str(ratio),'_acc_1to45_pca'];
save(path_orl,'acc','std_acc');
plot(1:45,acc);

