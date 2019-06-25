clc;
clear;

[Xu_imgs, Yu_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
[Xt_imgs, Yt_labels] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

% prepare data for network creation
% T = ...;


% create a Hopfield recurrent function
% net = newhop(T);
newhop();

