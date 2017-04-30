svm_template = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 'auto');

data_video_level_vid={};

for i=0:39
    data_video_level_vid{i+1} = csvread(['/Users/tarang/Documents/MLProject/video_level_csv/Class' num2str(i) '_vid.csv']);
end

training=[];
testing=[];
labels=[];

n=100;

for i = 0:39
    data = data_video_level_vid{i+1};
    training = [training;data(1:n,:)];
    testing = [testing;data(n+1:2*n,:)];
    labels = [labels;ones(n,1)+i];
end

svm_model = fitcecoc(training, labels, 'Learners', svm_template);
svm_predicted = predict(svm_model, testing);

count=0;
for i = 1:size(labels,1)
    if(labels(i)==svm_predicted(i))
        count=count+1;
    end
end

acc = count/length(labels)