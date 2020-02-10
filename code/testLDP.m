% �����������
clear
clc

% ��������
% ����Yale���ݼ�
% load('./���ݼ�/Yale_32x32.mat');
% classNum = 15;

% ����ORL���ݼ�
load('./���ݼ�/ORL_32x32.mat');
classNum = 40;

% ����YaleB���ݼ�
% load('./���ݼ�/YaleB_32x32.mat');
% classNum = 38;
ratio = 3;

for dim=1:45
    for i=1:10
        % ����ѵ���������Լ�
        [X_train, y_train, X_test, y_test] = Mysplit_train_test(fea, gnd, classNum, ratio);
        
        % ����MFA
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

