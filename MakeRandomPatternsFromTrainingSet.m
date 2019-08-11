function [ random_training_values ] = MakeRandomPatternsFromTrainingSet( training_set, training_labels, unique_labels )

    random_training_values=zeros(size(training_set, 1), size(training_set, 2), length(unique_labels));
    random_labels=zeros(size(training_set, 3), length(unique_labels));

    for i=1:size(training_set, 3)
        label=training_labels(i)+1;
        random_labels(i, label)=i;
    end

    for label=unique_labels
        label_index = find(random_labels(:,label));
        random_training_values(:,:,label)=training_set(:,:,label_index(randperm(length(label_index),1)));
    end

    random_training_values=reshape(random_training_values, [size(random_training_values, 1)*size(random_training_values, 2), size(random_training_values, 3)]);

end
