%% transfer learning with alexanet 

%% Analize net
net = alexnet;
%analyzeNetwork(net) 
inputSize = net.Layers(1).InputSize;

%% Load data
imageFolder_train='data/train';
imageFolder_test='data/test';
imds = imageDatastore(imageFolder_train, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
imds_test = imageDatastore(imageFolder_test, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%% Use net

%spilt data random in train and validation set
[trainingset, validationset]=splitEachLabel(imds, 0.85, 'randomize');

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

totdata = augmentedImageDatastore([227 227 3],imds, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');
augmentedTrainingSet = augmentedImageDatastore([227 227 3],trainingset, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');
augmentedValidationSet = augmentedImageDatastore([227 227 3],validationset, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');
augmentedTestSet = augmentedImageDatastore([227 227 3],imds_test, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');

%{
totdata = augmentedImageDatastore(inputSize, imds,'DataAugmentation',imageAugmenter);%, 'ColorPreprocessing', 'gray2rgb');

augmentedTrainingSet = augmentedImageDatastore(inputSize, trainingset, 'DataAugmentation',imageAugmenter);%, 'ColorPreprocessing', 'gray2rgb');
augmentedValidationSet = augmentedImageDatastore(inputSize, validationset, 'DataAugmentation',imageAugmenter);%, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(inputSize, imds_test, 'DataAugmentation',imageAugmenter);%, 'ColorPreprocessing', 'gray2rgb');
%}

    %% modifica layer
layersTransfer = net.Layers(1:end-3);
%freeze weight tranne untimi 3


numClasses = numel(categories(imds.Labels));


layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

layers(1:end-3)= freezeWeights(layers(1:end-3));

%% image set

augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationset);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedValidationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augmentedTrainingSet,layers,options);

predicted = classify(netTransfer,augmentedTestSet);
%%
accuracy = sum(predicted == imds_test.Labels)/numel(imds_test.Labels)

%[YPred,scores] = classify(netTransfer,augimdsValidation);

%idx = randperm(numel(imdsValidation.Files),4);


