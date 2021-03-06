%% init
clear

%% Costruzione della struttura della NN

layers = [
    imageInputLayer([64 64 1],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],16,"Name","conv_2","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([7 7],32,"Name","conv_3","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_3")
    dropoutLayer
    fullyConnectedLayer(15,"Name","fc",  "WeightsInitializer",'narrow-normal')
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];


%% Load data
imageFolder_train='data/train';
imageFolder_test='data/test';
imds = imageDatastore(imageFolder_train, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
imds_test = imageDatastore(imageFolder_test, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%spilt data random in train and validation set
[trainingset, validationset]=splitEachLabel(imds, 0.85, 'randomize');

%rescale images for learning
imageSize=[64 64 1];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',1, ...
    'RandYReflection',1, ...
    'RandRotation',[-5,5], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

totdata = augmentedImageDatastore(imageSize, imds,'DataAugmentation',imageAugmenter);%, 'ColorPreprocessing', 'gray2rgb');

augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingset, 'DataAugmentation',imageAugmenter);%, 'ColorPreprocessing', 'gray2rgb');
augmentedValidationSet = augmentedImageDatastore(imageSize, validationset, 'DataAugmentation',imageAugmenter);%, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, imds_test, 'DataAugmentation',imageAugmenter);%, 'ColorPreprocessing', 'gray2rgb');

%% Learn
%{
%'sgdm' 'adam' rmsprop
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',128,...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ValidationData',augmentedValidationSet);
net = trainNetwork(augmentedTrainingSet,layers,options);

%% Test
predicted = classify(net,augmentedTestSet)
plotconfusion(imds_test.Labels,predicted);
accuracy = sum(predicted == imds_test.Labels)/numel(imds_test.Labels)
predicted(2985)
%}

%% train for ensemble of networks
%train 5 to 10 reti e poi fare la media
trainig = 'training layers 1'
layers1 = [
    imageInputLayer([64 64 1],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],16,"Name","conv_2","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([7 7],32,"Name","conv_3","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_3")
    fullyConnectedLayer(15,"Name","fc",  "WeightsInitializer",'narrow-normal')
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

options1 = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',128,...
    'ValidationPatience',10,...
    'ValidationData',augmentedValidationSet);
net1 = trainNetwork(augmentedTrainingSet,layers1,options1);
predicted1 = classify(net1,augmentedTestSet);

trainig = 'training layers 2'
layers2 = [
    imageInputLayer([64 64 1],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],16,"Name","conv_2","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([7 7],32,"Name","conv_3","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_3")
    dropoutLayer
    fullyConnectedLayer(15,"Name","fc",  "WeightsInitializer",'narrow-normal')
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

options2 = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',128,...
    'ValidationData',augmentedValidationSet);
net2 = trainNetwork(augmentedTrainingSet,layers2,options2);
predicted2 = classify(net2,augmentedTestSet);

trainig = 'training layers 3'
layers3 = [
    imageInputLayer([64 64 1],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],16,"Name","conv_2","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([7 7],32,"Name","conv_3","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_3")
    dropoutLayer
    fullyConnectedLayer(15,"Name","fc",  "WeightsInitializer",'narrow-normal')
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

options3 = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',64,...
    'ValidationPatience',10,...
    'ValidationData',augmentedValidationSet);
net3 = trainNetwork(augmentedTrainingSet,layers3,options3);
predicted3 = classify(net3,augmentedTestSet);

trainig = 'training layers 4'
layers4 = [
    imageInputLayer([64 64 1],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],16,"Name","conv_2","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([7 7],32,"Name","conv_3","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same","Stride",[2 2])
    convolution2dLayer([9 9],32,"Name","conv_4","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_4")
    dropoutLayer
    fullyConnectedLayer(15,"Name","fc",  "WeightsInitializer",'narrow-normal')
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

options4 = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',64,...
    'ValidationPatience',10,...
    'ValidationData',augmentedValidationSet);
net4 = trainNetwork(augmentedTrainingSet,layers4,options4);
predicted4 = classify(net4,augmentedTestSet);

trainig = 'training layers 5'
layers5 = [
    imageInputLayer([64 64 1],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],16,"Name","conv_2","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([7 7],32,"Name","conv_3","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same","Stride",[2 2])
    convolution2dLayer([9 9],32,"Name","conv_4","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_4")
    maxPooling2dLayer([2 2],"Name","maxpool_4","Padding","same","Stride",[2 2])
    convolution2dLayer([11 11],64,"Name","conv_5","Padding","same", "WeightsInitializer",'narrow-normal')
    batchNormalizationLayer
    reluLayer("Name","relu_5")
    dropoutLayer
    fullyConnectedLayer(15,"Name","fc",  "WeightsInitializer",'narrow-normal')
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

options5 = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',128,...
    'ValidationPatience',10,...
    'ValidationData',augmentedValidationSet);
net5 = trainNetwork(augmentedTrainingSet,layers5,options5);
predicted5 = classify(net5,augmentedTestSet);

%% ensemble of networks 
a=predicted1(1);
b=predicted1(1);
c=predicted1(1);
d=predicted1(1);
e=predicted1(1);
a1=0;
b1=0;
c1=0;
d1=0;
e1=0;

for i = 1 : 2985
    a=predicted1(i);
    a1=1;
    if a==predicted2(i)
        a1=a1+1;
    else
        b=predicted2(i);
        b1=1;
    end
    
        
    if a==predicted3(i)
        a1=a1+1;
    elseif b==predicted3(i)
        b1=b1+1;
    else
        c=predicted2(i);
        c1=1;
    end
        
    if a==predicted4(i)
        a1=a1+1;
    elseif b==predicted4(i)
        b1=b1+1;
    elseif c==predicted4(i)
        c1=c1+1;
    else
        d=predicted2(i);
        d1=1;
    end
    
    if a==predicted5(i)
        a1=a1+1;
    elseif b==predicted5(i)
        b1=b1+1;
    elseif c==predicted5(i)
        c1=c1+1;
    elseif d==predicted5(i)
        d1=d1+1;
    else
        e=predicted2(i);
        e1=1;
    end
    
    val_max=max([a1,b1,c1,d1,e1]);
    
    if a1==val_max
        predicted(i)=a;
    elseif b1==val_max
        predicted(i)=b;
    elseif c1==val_max
        predicted(i)=c;
    elseif d1==val_max
        predicted(i)=d;
    elseif e1==val_max
        predicted(i)=e;
    end
    
end
predicted_test=predicted';
randiom=imds_test.Labels;
accuracy = sum(predicted_test == imds_test.Labels)/numel(imds_test.Labels)


