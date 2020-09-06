%% Costruzione della struttura della NN

layers = [
    imageInputLayer([64 64 1],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same", "WeightsInitializer",'narrow-normal')
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_2","Padding","same", "WeightsInitializer",'narrow-normal')
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_3","Padding","same", "WeightsInitializer",'narrow-normal')
    reluLayer("Name","relu_3")
    fullyConnectedLayer(15,"Name","fc",  "WeightsInitializer",'narrow-normal')
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];


%% Load data
imageFolder_train='data/train';
imageFolder_test='data/test';
imds = imageDatastore(imageFolder_train, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
imds_test = imageDatastore(imageFolder_test, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%% plot test
%daisy = find(imds.Labels == 'Bedroom', 1);
%figure
%imshow(readimage(imds,daisy))

%% verifica la distribuzione delle foto
%tbl = countEachLabel(imds)

%% se il set non è uniforme run this
% Determine the smallest amount of images in a category
%minSetCount = min(tbl{:,2}); 

% Limit the number of images to reduce the time it takes
% run this example.
%maxNumImages = 100;
%minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
%imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
%countEachLabel(imds)

%% Use net
%test_ima=imresize(test,[64 64])
%imageSize=net.Layers(1).InputSize;%output dovrebbe essere [64 64]

%spilt data random in train and validation set
[trainingset, validationset]=splitEachLabel(imds, 0.85, 'randomize');

%rescale images for learning
imageSize=[64 64 1];%imageSize=net.Layers(1).InputSize;%output dovrebbe essere [64 64]

totdata = augmentedImageDatastore(imageSize, imds);%, 'ColorPreprocessing', 'gray2rgb');

augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingset);%, 'ColorPreprocessing', 'gray2rgb');
augmentedValidationSet = augmentedImageDatastore(imageSize, validationset);%, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, imds_test);%, 'ColorPreprocessing', 'gray2rgb');

%% Learn

options = trainingOptions('sgdm', ...
    'MiniBatchSize',32,...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ValidationData',{augmentedTrainingSet,augmentedValidationSet});
net = trainNetwork(totdata,layers,options);

%% Test
predicted = classify(net,augmentedTestSet);
%plotconfusion(imds_test.Labels,predicted);
accuracy = sum(predicted == imds_test.Labels)/numel(imds_test.Labels)

%[XTest,~,YTest] = digitTest4DArrayData;
%YPred = predict(net,XTest);
%predictionError = YTest - YPred;
%thr = 10;
%numCorrect = sum(abs(predictionError) < thr);
%numTestImages = size(XTest,4);
%accuracy = numCorrect/numTestImages


