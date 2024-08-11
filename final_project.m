clear 
% load data
imds = imageDatastore('WaveformDataset\','IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.mat'});
% plit data into 3 subsets
[imdsTrain,imdsTest,imdsVal] = splitEachLabel(imds,0.7,0.15,'randomized');
% create the datastorage 
imdsTrain.Labels    = categorical(imdsTrain.Labels);imdsTrain.ReadFcn = @readFcnMatFile;
imdsTest.Labels     = categorical(imdsTest.Labels);imdsTest.ReadFcn = @readFcnMatFile;
imdsVal.Labels      = categorical(imdsVal.Labels);imdsVal.ReadFcn = @readFcnMatFile;

% training options configuration
network_graph; % load network architecture

% set up training parameters
maxepoch = 40; % fix it (students do not change it)

batchSize   = 32; % batchsize and other hyper-parameters can be modified
ValFre      = fix(length(imdsTrain.Files)/batchSize);
options = trainingOptions('adam', ...
    'MiniBatchSize',batchSize, ...
    'MaxEpochs',maxepoch, ...
    'Shuffle','every-epoch',...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.2,...
    'ValidationData',imdsVal, ...
    'ValidationFrequency',ValFre, ...
    'ValidationPatience',20, ...
    'Verbose',true ,...
    'VerboseFrequency',ValFre,...
    'Plots','training-progress',...
    'ExecutionEnvironment','auto',...
    'OutputNetwork','best-validation-loss');

% train the model with the imdsTrain set and validate with the imdsVal set.
trainednet = trainNetwork(imdsTrain,lgraph,options);


% the following is the performance evaluation
% require the students do not change
YPred = classify(trainednet,imdsTest,'MiniBatchSize',128,'ExecutionEnvironment','cpu');
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest)
trainednetInfo = {};
trainednetInfo{1,1} = trainednet;
trainednetInfo{1,2} = YTest;
trainednetInfo{1,3} = YPred;
trainednetInfo{1,4} = accuracy;
trainednetInfo{1,5} = imdsTrain;
trainednetInfo{1,6} = imdsTest;
save('trainednet.mat','trainednetInfo')


