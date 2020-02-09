% 清除环境
clear all;
clc;

% 样本个数
ratio = 5;

% path = 'ORL_L';
path = 'Yale_L';

% 绘制曲线图
path_pca = [path, num2str(ratio), '_acc_1to45_pca'];
pca = load(path_pca,'acc');
path_mmc = [path,num2str(ratio),'_acc_1to45_mmc'];
mmc = load(path_mmc,'acc');
path_mfa = [path,num2str(ratio),'_acc_1to45_mfa'];
mfa = load(path_mfa,'acc');
path_mdp = [path,num2str(ratio),'_acc_1to45_mdp'];
mdp = load(path_mdp,'acc');

% 取结构体中的值
Yale_pca = getfield(pca,'acc');
Yale_mmc = getfield(mmc,'acc');
Yale_mfa = getfield(mfa,'acc');
Yale_mdp = getfield(mdp,'acc');

x = 1:45;
plot(x,Yale_pca,':^',x,Yale_mmc,':V',x,Yale_mfa,':*',x,Yale_mdp,':>');
xlabel('维数');
ylabel('准确率');
legend('PCA','MMC','MFA','MDP','Location','Best');

[Y_pca,I_pca] = max(Yale_pca)
[Y_mmc,I_mmc] = max(Yale_mmc)
[Y_mfa,I_mfa] = max(Yale_mfa)
[Y_mdp,I_mdp] = max(Yale_mdp)

% % 显示图片
% load('./数据集/Yale_32x32.mat');
% faceW = 32; 
% faceH = 32; 
% numPerLine = 11; 
% ShowLine = 7; 
% 
% Y = zeros(faceH*ShowLine,faceW*numPerLine); 
% for i=0:ShowLine-1 
%   	for j=0:numPerLine-1 
%     	Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(fea(i*numPerLine+j+1,:),[faceH,faceW]); 
%   	end 
% end 
% 
% imagesc(Y);colormap(gray);

