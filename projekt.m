clc;
clear;

% https://www.alexbod.com/hopfield-neural-network/#matlab

[Xu_imgs, Yu_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
[Xt_imgs, Yt_labels] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);
% 
% net = newhop();
% T=Xu_imgs;
% T=reshape(Xu_imgs, [size(Xu_imgs, 1)*size(Xu_imgs, 2), size(Xu_imgs, 3)]);
% T=T';
v=unique(Yu_labels)';
T=zeros(size(Xu_imgs, 1), size(Xu_imgs, 2),length(v));
c=1;
for l=v
    i=1;
    while Yu_labels(i)~=l
        i=i+1;
    end
    T(:,:,c)=Xu_imgs(:,:,i);
    c=c+1;
end
T=reshape(T, [size(T, 1)*size(T, 2), size(T, 3)]);
% T=Xu_imgs(:,:,1);
% T = [-1 -1 1; 1 -1 1]';
net = newhop(T);
Ai = T;
[Y,Pf,Af] = net(size(Ai, 2),[],Ai);

sum(sum(abs(Y-T)))

% Ai = Yu_labels;
% % Ai = {[-0.9; -0.8; 0.7]};
% [Y,Pf,Af] = net({1 5},{},Ai);
% Y{1}
