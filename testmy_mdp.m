% 清除环境变量
clear
clc

% 加载Yale数据集
load('./数据集/Yale_32x32.mat');
classNum = 15;

% % 加载ORL数据集
% load('./数据集/ORL_32x32.mat');
% classNum = 40;

% % 加载YaleB数据集
% load('./数据集/YaleB_32x32.mat');
% classNum = 38;
ratio = 4;

for dim=1:45
    for i=1:10
        % 划分训练集，测试集
        [X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, classNum, ratio);
        
        % 测试MDP
        k = dim;
        k1 = 4;
        k2 = 1;
%         [W] = myMDP(y_train, k, X_train, k1, k2);
        [W,~] = my_MDP(X_train,y_train, k);
        X_train_mdp = X_train*W;
        X_test_mdp = X_test*W;
        
        accuracy(i) = KNN(X_train_mdp,y_train,X_test_mdp,y_test,1);
    end
    acc(dim) = mean(accuracy);
    std_acc(dim) = std(accuracy);
end

% acc = mean(accuracy);
% std = std(accuracy);

path = ['Yale_L',num2str(ratio),'_acc_1to45_hjr_mdp'];
save(path,'acc','std_acc');
plot(1:45,acc);


