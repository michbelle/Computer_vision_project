%% transfer learning with alexanet 
%analyzeNetwork(net) 
%analyzeNetwork(netTransfer) 
clear
%% Analize net
net = alexnet;
analyzeNetwork(net) 
inputSize = net.Layers(1).InputSize;

%% Load data
imageFolder_train='data/train';
imageFolder_test='data/test';
imds = imageDatastore(imageFolder_train, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
imds_test = imageDatastore(imageFolder_test, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%% setup data

%spilt data random in train and validation set
[trainingset, validationset]=splitEachLabel(imds, 0.85, 'randomize');

pixelRange = [-3 3];
scaleRange = [1 1.3];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandRotation',[-5,5], ...
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

%% modifica layer
    
layersTransfer = net.Layers(1:end-3);
%freeze weight tranne untimi 3


numClasses = numel(categories(imds.Labels));
%WI:'glorot'(default)'he''orthogonal''narrow-normal''zeros''ones' 
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, ...
    'WeightsInitializer', 'glorot', ...
    'BiasInitializer','zeros', ...
    'WeightLearnRateFactor',40, ...
    'BiasLearnRateFactor',20,...
    'WeightL2Factor',1,...
    'BiasL2Factor',0)
    softmaxLayer
    classificationLayer];

layers(1:end-3)= freezeWeights(layers(1:end-3));

%% image set

augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationset);
% solverName — Solver for training network 'sgdm' | 'rmsprop' | 'adam'
options = trainingOptions('adam', ...
    'MiniBatchSize',256, ... %128
    'MaxEpochs',100, ...
    'InitialLearnRate',0.00005, ... %1e-4
    'Shuffle','every-epoch', ...
    'ValidationPatience',30,... %Inf
    'ValidationData',augmentedValidationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augmentedTrainingSet,layers,options);

predicted = classify(netTransfer,augmentedTestSet);
%%
accuracy = sum(predicted == imds_test.Labels)/numel(imds_test.Labels);
disp('accuracy tranfer learning')
disp(accuracy)

%% extraction feature 

featureLayer ='conv5';
trainingFeatures = activations(netTransfer, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 256, 'OutputAs', 'columns');

featuresTest = activations(netTransfer,augmentedTestSet,featureLayer,...
    'MiniBatchSize', 256, 'OutputAs', 'rows');

% Get training labels from the trainingSet
trainingLabels = trainingset.Labels;
YTest = imds_test.Labels;

%% Svm da matlab

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.

%
t = templateSVM('Standardize',true,'KernelFunction','Linear');

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', t , 'Coding', 'onevsone', 'ObservationsIn', 'columns');
%}

%{
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
%}
YPred = predict(classifier,featuresTest);

accuracy = sum(YPred == imds_test.Labels)/numel(imds_test.Labels);
disp('accuracy SVM:')
disp(accuracy)


