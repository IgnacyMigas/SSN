clc;
clear;

%zbiór Uczacy
[Xu_imgs, Yu_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
%zbiór Testowy
[Xt_imgs, Yt_labels] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

%normalizacja do wartoœci [0,1]
Xu_imgs=normalizePixValue(Xu_imgs);
Xt_imgs=normalizePixValue(Xt_imgs);

%unikatowe etykiety dla zbioru ucz¹cego (0,1,2,...9)
%+1 aby wszystkie wartoœci by³y dodatnie
unique_labels=unique(Yu_labels)'+1;

%T - uœredniony zbiór ucz¹cy
T=MakeAveragedPatternsFromTrainingSet(Xu_imgs, Yu_labels, unique_labels);

%tworzenie nowej sieci Hopfielda
net = newhop(T);

noise_level = [0,1,5,10,15,20,30,40,50,60,70,80];
%przygotowanie zbioru testowego
Xt_dissorted_all = AddNoiseToAll(Xt_imgs, noise_level(1));
Xt_dissorted_random = AddNoiseToRandomElements(Xt_imgs, noise_level(1));
Xt = reshape(Xt_imgs, [size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3)]);

%odpowiedŸ sieci na zbiór testowy
Y = net(size(Xt, 2),[],Xt);

%porównanie wyników z szablonami ze zbioru ucz¹cego
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

%ró¿nica miêdzy zbiorem testowym, a odpowiedzi¹ sieci
%(przyporz¹dkowanymi szablonami) wyra¿ona w procentach
text1='ró¿nica miêdzy zbiorem testowym, a odpowiedzi¹ sieci (%)';
disp(text1);
disp(sum(sum(abs(Y-Xt)))/size(Y, 2));

%zliczenie b³êdnych przyporz¹dkowañ
s=nnz(Y_labels-Yt_labels);
text2='liczba b³êdnych przyporz¹dkowañ';
disp(text2);
disp(s);
text3='liczba przyporz¹dkowañ';
disp(text3);
disp(size(Xt, 2));

fprintf('Accuracy = %f%%\n', (1-s/length(Y_labels))*100);
fprintf('Error = %f%%\n', s/length(Y_labels)*100);
