% 清除环境变量
clear
clc

% 加载数据
% 加载Yale数据集
% load('./数据集/Yale_32x32.mat');
% classNum = 15;

% 加载ORL数据集
load('./数据集/ORL_32x32.mat');
classNum = 40;

% 加载YaleB数据集
% load('./数据集/YaleB_32x32.mat');
% classNum = 38;
ratio = 3;

for dim=1:45
    for i=1:10
        % 划分训练集，测试集
        [X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, classNum, ratio);
        
        % 测试MFA
        k = dim;
        k1 = 1;
        k2 = 1;
        [W] = LDP(y_train, k, X_train, k1, k2);
        if k > 29
            k = 29;
        end
        X_train_mfa = X_train*W(:,1:k);
        X_test_mfa = X_test*W(:,1:k);
        accuracy(i) = KNN(X_train_mfa,y_train,X_test_mfa,y_test,5);
    end
    acc(dim) = mean(accuracy);
    std_acc(dim) = std(accuracy);
end


path = ['Yale_L',num2str(ratio),'_acc_1to45_ldp'];
path_orl = ['ORL_L',num2str(ratio),'_acc_1to45_ldp'];
save(path_orl,'acc','std_acc');
plot(1:45,acc);

