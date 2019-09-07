function [ averaged_training_values ] = MakeAveragedPatternsFromTrainingSet( training_set, training_labels, unique_labels )

averaged_training_values=zeros(size(training_set, 1), size(training_set, 2), length(unique_labels));
label_count=zeros(length(unique_labels), 1);

for i=1:size(training_set, 3)
    l=training_labels(i)+1;
    averaged_training_values(:,:,l)=averaged_training_values(:,:,l)+training_set(:,:,i);
    label_count(l)=label_count(l)+1;
end

for label=unique_labels
    averaged_training_values(:,:,label)=averaged_training_values(:,:,label)/label_count(label);
end

averaged_training_values=reshape(averaged_training_values, [size(averaged_training_values, 1)*size(averaged_training_values, 2), size(averaged_training_values, 3)]);

end

