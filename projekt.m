% clc;
% clear;

% % % przykład aproksymacji
% spirala fermata
% spirala logarytmiczna

% % % przykład klasyfikacji
% load iris_dataset
% % może drugi przykład klasyfikacji
% load dane_medyczne/Learning.mat;
% load dane_medyczne/Test.mat

% % spirala logarytmiczna
% a=1;
% b=0.178;
% Fi=0:0.3:pi*5;
% r=a*exp(b*Fi);
% x=r.*cos(Fi);
% y=r.*sin(Fi);
% plot(x, y);

% % spirala fermata
% dane uczace
a=2.5;
Fi=pi*5:-0.3:0;
r=sqrt(a*a*Fi);
tmp=Fi(end:-1:1);
r=[r -sqrt(a*a*tmp)];
Fi=[Fi tmp];
x=r.*cos(Fi);
y=r.*sin(Fi);
clear tmp;
plot(x, y);
len=size(x,2);
% % dane testowe
Fi=pi*5:-0.5:0;
r=sqrt(a*a*Fi);
tmp=Fi(end:-1:1);
r=[r -sqrt(a*a*tmp)];
Fi=[Fi tmp];
xt=r.*cos(Fi);
yt=r.*sin(Fi);
clear tmp;
% figure, plot(xt, yt);

% len=10;
% inp=zeros(len+1, 1);
% for i=1:len
%     inp(i)=1;
% end
% % inp(1,1)=0;
% net=network(1, len+1);
% % net.layerConnect(2,1)=1;
% % net.layers{1}.size=len;
% 
% for i=1:len
%     net.layers{i}.size=1;
%     net.layers{i}.transferFcn = 'radbas';
%     net.layerConnect(len+1,i)=1;
% end
% net.layers{len+1}.size=1;
% net.inputConnect=inp;
% net.outputConnect(len+1)=1;
% view(net);

% len=10;
% inp=[ones(1, len); zeros(1, len)];
% % inp(1,1)=0;
% net=network(len, 2);
% for i=1:len
%     net.inputs{i}.size=1;
% end
% net.layerConnect(2,1)=1;
% net.layers{1}.size=len;
% net.layers{1}.transferFcn = 'radbas';
% net.biasConnect(1)=1;
% net.layers{2}.size=1;
% net.inputConnect=inp;
% net.outputConnect(2)=1;
% view(net);

inp=[1;0];%[ones(1, len); zeros(1, len)];
% inp(1,1)=0;
net=network(1,2, [1;1], inp, [0,0;1,0], [0,1]);

net.inputs{1}.size=1;
% % % 
% 
%   net.inputWeights{1,1}.weightFcn = 'dist';
%   net.layers{1}.netInputFcn = 'netprod';
%   net.layers{1}.transferFcn = 'radbas';
%   net.layers{2}.size = S2;
%   net.outputs{2}.exampleOutput = t;
%   
%   % Performance
%   net.performFcn = 'mse';
% % % 

net.layers{1}.size=len;
net.layers{1}.transferFcn = 'radbas';
% net.layers{1}.distanceFcn = 'dist';% boxdist, dist, linkdist, mandist
net.layers{1}.initFcn = 'initwb';
net.layers{2}.size=1;
net.trainFcn='trainlm'; % algorytm gini'ego zamiast
net.trainParam.epochs=100;

% view(net);

net=init(net);
net=train(net, x, y); % funkcja do uczenia
T=net(xt);
plot(xt, T);
% % 
% % x=[x;x];
% % 
% % load iris_dataset
% % 
% % x = irisInputs;
% % y = irisTargets;
% % 
% % e=10;
% % sc=10;
% % net=newrb(x, y, e, sc);
% % net.trainFcn='trainlm';
% % net=train(net, x, y);
% % Y=round(net(x));
% % 
% % % disp(mse(Y, y));
% % % figure, plot(x, Y);
% % plotconfusion(Y, y);

