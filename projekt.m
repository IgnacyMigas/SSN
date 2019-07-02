clc;
clear;

% https://www.alexbod.com/hopfield-neural-network/#matlab

[Xu_imgs, Yu_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
[Xt_imgs, Yt_labels] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

Xu_imgs=normalizePixValue(Xu_imgs);
Xt_imgs=normalizePixValue(Xt_imgs);

unique_labels=unique(Yu_labels)'+1;

T=zeros(size(Xu_imgs, 1), size(Xu_imgs, 2),length(unique_labels));
T_cnt=zeros(length(unique_labels), 1);

for i=1:size(Xu_imgs, 3)
    l=Yu_labels(i)+1;
    T(:,:,l)=T(:,:,l)+Xu_imgs(:,:,i);
    T_cnt(l)=T_cnt(l)+1;
end

for label=unique_labels
    T(:,:,label)=T(:,:,label)/T_cnt(label);
end

T=reshape(T, [size(T, 1)*size(T, 2), size(T, 3)]);
net = newhop(T);

Xt = reshape(Xt_imgs, [size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3)]);

[Y,Pf,Af] = net(size(Xt, 2),[],Xt);

disp(sum(sum(abs(Y-Xt)))/size(Y, 2))

