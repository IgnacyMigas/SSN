clc;
clear;

%%% training data
[Xu_imgs, Yu_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
%%% testing data
[Xt_imgs, Yt_labels] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

%%% prepare data for network creation
%%% T = ...;

% xx = Xu_imgs(:,:,1) % checking

%%% create a Hopfield recurrent network
%%% net = newhop(T);
% net = newhop();

