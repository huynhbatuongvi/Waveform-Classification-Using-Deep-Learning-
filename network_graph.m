lgraph = layerGraph();

tempLayers = [
    imageInputLayer([64 64 9],"Name","input_1")
    convolution2dLayer([7 7],64,"Name","conv","Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm","Epsilon",0.001)
    reluLayer("Name","relu")
    maxPooling2dLayer([3 3],"Name","max_pooling","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","conv_1")
    batchNormalizationLayer("Name","batchnorm1","Epsilon",0.001)
    reluLayer("Name","relu1")
    convolution2dLayer([3 3],32,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm2","Epsilon",0.001)
    reluLayer("Name","relu2")
    convolution2dLayer([1 1],128,"Name","conv_3")
    batchNormalizationLayer("Name","batchnorm3","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv_4")
    batchNormalizationLayer("Name","batchnorm4","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    reluLayer("Name","relu3")
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(8,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"max_pooling","conv_1");
lgraph = connectLayers(lgraph,"max_pooling","conv_4");
lgraph = connectLayers(lgraph,"batchnorm3","addition/in1");
lgraph = connectLayers(lgraph,"batchnorm4","addition/in2");