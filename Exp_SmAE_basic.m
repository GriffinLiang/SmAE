% %% load data
load('D:\Dataset\Variations on MNIST\mnist_basic.mat') ;

valLabels(valLabels == 0) = 10 ;
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
testLabels(testLabels == 0) = 10;

numClasses = size(unique(trainLabels),2) ;
inputSize = size(trainData, 1) ;
n_trainData = size(trainData, 2) ;
n_testData = size(testData, 2) ;

%% 1. initialize parameter
%%%% network
layersizes = [inputSize 1000 ] ;
sparsityParamSet = 0.1;  %% 0.1 
lambdaSet = 0 ;            
betaSet = 0 ;  % 1 0.1 0.01 0.001  
sparsityParam = sparsityParamSet ;
lambda = lambdaSet ;
beta = betaSet ;
%%%% optimization
maxIter = 10 ;
batchSize = n_trainData ;
optionsAE.Method = 'cg'; 
optionsAE.maxIter = 400 ;	
optionsAE.display = 'on';
%%%% classifier
c_lambda = 1e-4 ;
ft_lambda = lambda ;
%%%% usage
global useGpu
global tiedWeight;

useGpu = false ;
tiedWeight = true ;

%%%% cross validation
svNumSet = size(trainData, 2) ;
svNum = svNumSet ;
%%%% result 
acc = 0 ;
acc_FT = 0 ;
n_round = 3 ;
%%%% SmRelationship
k_smooth = 10 ;
guassBeta = 0.1 ;
% [knnIdx, dist] = knnsearch(trainData', trainData', 'k', k_smooth) ;
% idxSm = knnIdx ;
% coef = bsxfun(@rdivide, exp(-dist*guassBeta), sum(exp(-dist*guassBeta),2));
for cc = 1:numClasses
    index = find(trainLabels == cc) ;
    [knnIdx, dist] = knnsearch(trainData(:,index)', trainData(:,index)', 'k', k_smooth) ;
    idxSm(index,:) = index(knnIdx) ;
    coef(index,:) = bsxfun(@rdivide, exp(-dist*guassBeta), sum(exp(-dist*guassBeta),2));
end

for cc = 1:numClasses
    index = find(testLabels == cc) ;
    [knnIdx, dist] = knnsearch(testData(:,index)', testData(:,index)', 'k', k_smooth) ;
    idxSm_test(index,:) = index(knnIdx) ;
    coef_test(index,:) = bsxfun(@rdivide, exp(-dist*guassBeta), sum(exp(-dist*guassBeta),2));
end

addpath D:\Code\UMF\minFunc\
%% 2. layerwise pretrain
input = trainData ;
theta = [] ;
for num_layer = 1:(length(layersizes)-1)
    saeTheta{num_layer} = initializeParameters(layersizes(num_layer+1), layersizes(num_layer));
    smData = sum(reshape(bsxfun(@times, input(:, idxSm(:)), coef(:)'),layersizes(num_layer),...
                 n_trainData, k_smooth),3) ;
    ae.visibleSize = layersizes(num_layer);
    ae.hiddenSize = layersizes(num_layer+1);
    ae.lambda = lambda;
    ae.sparsityParam = sparsityParam ;
    ae.beta = beta ;
    [saeOptTheta{num_layer}, cost] = minFunc( @(p) smoothsparseTiedAECrossEntropyCost(p, ...
                                input, smData, ae), saeTheta{num_layer}, optionsAE);

    [input] = feedForwardAutoencoder(saeOptTheta{num_layer}, layersizes(num_layer+1), ...
                                            layersizes(num_layer), input);                           
end

output = input ;

%% 3. all layers pretrain
theta1 = [] ;           % strech the theta to the form: [ W1 b1 W2 b2 .... b2' b1']
theta2 = [] ;
for num_layer = 1:(length(layersizes)-1)
    if tiedWeight
        theta1 = [theta1 ; saeOptTheta{num_layer}(1:end-layersizes(num_layer)) ] ; 
    else
        theta1 = [theta1 ; saeOptTheta{num_layer}(1:layersizes(num_layer)*layersizes(num_layer+1)) ] ;
        theta1 = [theta1 ; saeOptTheta{num_layer}(end-layersizes(num_layer)-...
                            layersizes(num_layer+1)+1:end-layersizes(num_layer)) ] ;
    end
    theta2 = [saeOptTheta{num_layer}(end-layersizes(num_layer)+1:end) ; theta2] ;
end
theta = [theta1; theta2] ;
stackTheta = theta1 ;


options.Method = 'cg'; 
options.maxIter = 200 ;	
options.display = 'on';


%% 4. layerwise classifier
saeSoftmaxTheta = 0.005 * randn(layersizes(end) * numClasses, 1);

softmaxModel = softmaxTrain(layersizes(end), numClasses, c_lambda, ...
                        output(:, 1:svNum), trainLabels(1:svNum), options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% %% 5. Fine-tune
stackedAETheta = [ saeSoftmaxOptTheta ; stackTheta ];

[stackedAEOptTheta, cost] = minFunc( @(p) FinetuneAECost(p, layersizes,numClasses,...
                                ft_lambda, trainData, trainLabels),stackedAETheta, options);

%% 6. test
[pred] = stackedAEPredict(stackedAETheta, layersizes, numClasses, testData);
acc = mean(testLabels(:) == pred(:));
[pred_FT] = stackedAEPredict(stackedAEOptTheta, layersizes, numClasses, testData);
acc_FT = mean(testLabels(:) == pred_FT(:));
confusion_matrix(pred_FT, testLabels) ;