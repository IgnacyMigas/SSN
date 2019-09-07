clc;
clear;

%zbiór Uczacy
[Xu_imgs, Yu_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
%zbiór Testowy
[Xt_imgs, Yt_labels] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);

%normalizacja do wartoœci z przedzia³u [0,1]
Xu_imgs=normalizePixValue(Xu_imgs);
Xt_imgs=normalizePixValue(Xt_imgs);

%unikatowe etykiety dla zbioru ucz¹cego (0,1,2,...9)
%+1 aby wszystkie wartoœci by³y dodatnie
unique_labels=unique(Yu_labels)'+1;

%T - uœredniony zbiór ucz¹cy
T=MakeAveragedPatternsFromTrainingSet(Xu_imgs, Yu_labels, unique_labels);

%tworzenie nowej sieci Hopfielda
net = newhop(T);

noise_level = [0,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8];
n=size(noise_level, 2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%przygotowanie zbioru testowego

Xt_dissorted_all = zeros(size(Xt_imgs, 1), size(Xt_imgs, 2), size(Xt_imgs, 3), size(noise_level, 2));
Xt_dissorted_random = zeros(size(Xt_imgs, 1), size(Xt_imgs, 2), size(Xt_imgs, 3), size(noise_level, 2));

for m=1:n
    Xt_dissorted_all(:,:,:,m) = AddNoiseToAll(Xt_imgs, noise_level(m));
    Xt_dissorted_random(:,:,:,m) = AddNoiseToRandomElements(Xt_imgs, noise_level(m));
end

X_dissorted_all = zeros(size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3), n);
X_dissorted_random = zeros(size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3), n);

for m=1:n
    X_dissorted_all(:,:,m) = reshape([Xt_dissorted_all(:,:,:,m)], [size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3)]);
    X_dissorted_random(:,:,m) = reshape([Xt_dissorted_random(:,:,:,m)], [size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3)]);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%odpowiedŸ sieci na zbiór testowy

Y_dissorted_all = zeros(size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3), n);
Y_dissorted_random = zeros(size(Xt_imgs, 1)*size(Xt_imgs, 2), size(Xt_imgs, 3), n);

for m=1:n
    Y_dissorted_all(:,:,m) = net(size(X_dissorted_all(:,:,m), 2),[],X_dissorted_all(:,:,m));
    Y_dissorted_random(:,:,m) = net(size(X_dissorted_random(:,:,m), 2),[],X_dissorted_random(:,:,m));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%porównanie wyników z szablonami ze zbioru ucz¹cego

Y_dissorted_all_labels = zeros(size(Y_dissorted_all, 2), n);
Y_dissorted_random_labels = zeros(size(Y_dissorted_random, 2), n);

for m=1:n
    for i=1:size(Y_dissorted_all(:,:,m), 2)
        final_label_all=-1;
        final_label_random=-1;
        min_diff_all=-1;
        min_diff_random=-1;
        for j=1:size(T, 2)
            diff_all=sum(abs(Y_dissorted_all(:,i,m)-T(:,j)));
            if min_diff_all == -1 || diff_all < min_diff_all
                min_diff_all=diff_all;
                final_label_all=j-1;
            end
            diff_random=sum(abs(Y_dissorted_random(:,i,m)-T(:,j)));
            if min_diff_random == -1 || diff_random < min_diff_random
                min_diff_random=diff_random;
                final_label_random=j-1;
            end
        end
        Y_dissorted_all_labels(i,m)=final_label_all;
        Y_dissorted_random_labels(i,m)=final_label_random;

    end
end


%ró¿nica miêdzy zbiorem testowym, a odpowiedzi¹ sieci
%(przyporz¹dkowanymi szablonami) wyra¿ona w procentach
%dla poszczególnych zak³óceñ

for m=1:n
    disp('ró¿nica miêdzy zbiorem testowym, a odpowiedzi¹ sieci (%)');
    fprintf('zaburzenie = %d%%\n', noise_level(m)*100);
    
    disp('dodane do wszystkich elementów');
    disp(sum(sum(abs(Y_dissorted_all(:,:,m)- X_dissorted_all(:,:,m))))/size(Y_dissorted_all(:,:,m), 2));
    
    disp('dodane do losowych elementów');
    disp(sum(sum(abs(Y_dissorted_random(:,:,m)- X_dissorted_random(:,:,m))))/size(Y_dissorted_random(:,:,m), 2));
end


%zliczenie b³êdnych przyporz¹dkowañ

s_dissorted_all = zeros(n,1);
s_dissorted_random = zeros(n,1);

for m=1:n
    s_dissorted_all(m) = nnz(Y_dissorted_all_labels(:,m) - Yt_labels);
    s_dissorted_random(m) = nnz(Y_dissorted_random_labels(:,m) - Yt_labels);
    disp('liczba b³êdnych przyporz¹dkowañ');
    fprintf('zaburzenie = %d%%\n', noise_level(m)*100);
    
    disp('dodane do wszystkich elementów');
    disp(s_dissorted_all(m));
    
    disp('dodane do losowych elementów');
    disp(s_dissorted_random(m));
end

%wyninki koñcowe

for m=1:n
    fprintf('zaburzenie = %d%%\n', noise_level(m)*100);
    disp('dodane do wszystkich elementów');
    fprintf('Poprawne przyporz¹dkowania = %f%%\n', (1-s_dissorted_all(m)/length(Y_dissorted_all_labels(:,m)))*100);
    fprintf('B³êdne przyporz¹dkowania = %f%%\n', s_dissorted_all(m)/length(Y_dissorted_all_labels(:,m))*100);
    
    fprintf('zaburzenie = %d%%\n', noise_level(m)*100);
    disp('dodane do losowych elementów');
    fprintf('Poprawne przyporz¹dkowania = %f%%\n', (1-s_dissorted_random(m)/length(Y_dissorted_random_labels(:,m)))*100);
    fprintf('B³êdne przyporz¹dkowania = %f%%\n', s_dissorted_random(m)/length(Y_dissorted_random_labels(:,m))*100);
    
    fprintf('\n');
end
