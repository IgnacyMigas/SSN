clc;
clear;

%zbi�r Uczacy
[Xu_imgs, Yu_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
%zbi�r Testowy
[Xt_imgs, Yt_labels] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

%normalizacja do warto�ci [0,1]
Xu_imgs=normalizePixValue(Xu_imgs);
Xt_imgs=normalizePixValue(Xt_imgs);

%unikatowe etykiety dla zbioru ucz�cego (0,1,2,...9)
%+1 aby wszystkie warto�ci by�y dodatnie
unique_labels=unique(Yu_labels)'+1;

%T - u�redniony zbi�r ucz�cy
T=MakeAveragedPatternsFromTrainingSet(Xu_imgs, Yu_labels, unique_labels);

%tworzenie nowej sieci Hopfielda
net = newhop(T);

noise_level = [0,1,5,10,15,20,30,40,50,60,70,80];
%przygotowanie zbioru testowego
Xt_dissorted_all = AddNoiseToAll(Xt_imgs, noise_level(1));
Xt_dissorted_random = AddNoiseToRandomElements(Xt_imgs, noise_level(1));
Xt = reshape(Xt_imgs, [size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3)]);

%odpowied� sieci na zbi�r testowy
Y = net(size(Xt, 2),[],Xt);

%por�wnanie wynik�w z szablonami ze zbioru ucz�cego
Y_labels = zeros(size(Y, 2),1);
for i=1:size(Y, 2)
    final_label=-1;
    min_diff=-1;
    for j=1:size(T, 2)
        diff=sum(abs(Y(:,i)-T(:,j)));
        if min_diff == -1 || diff < min_diff
            min_diff=diff;
            final_label=j-1;
        end
    end
    Y_labels(i)=final_label;
end

%r�nica mi�dzy zbiorem testowym, a odpowiedzi� sieci
%(przyporz�dkowanymi szablonami) wyra�ona w procentach
text1='r�nica mi�dzy zbiorem testowym, a odpowiedzi� sieci (%)';
disp(text1);
disp(sum(sum(abs(Y-Xt)))/size(Y, 2));

%zliczenie b��dnych przyporz�dkowa�
s=nnz(Y_labels-Yt_labels);
text2='liczba b��dnych przyporz�dkowa�';
disp(text2);
disp(s);
text3='liczba przyporz�dkowa�';
disp(text3);
disp(size(Xt, 2));

fprintf('Accuracy = %f%%\n', (1-s/length(Y_labels))*100);
fprintf('Error = %f%%\n', s/length(Y_labels)*100);
