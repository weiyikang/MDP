% �����������
clear
clc

% ����Yale���ݼ�
load('./���ݼ�/Yale_32x32.mat');
classNum = 15;

% % ����ORL���ݼ�
% load('./���ݼ�/ORL_32x32.mat');
% classNum = 40;

% % ����YaleB���ݼ�
% load('./���ݼ�/YaleB_32x32.mat');
% classNum = 38;
ratio = 4;

for dim=1:45
    for i=1:10
        % ����ѵ���������Լ�
        [X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, classNum, ratio);
        
        % ����MDP
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


