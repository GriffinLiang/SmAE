%% load data
load('Yale_50_50.mat') ; 
data = [trainData testData] ;
label = [trainLabels testLabels] ;
trainData = [] ;
trainLabels = [] ;
testData = [] ;
testLabels = [] ;
for person = 1:38
    personData = data(:, label == person) ;
    num_person = size(personData, 2) ;
    perm = randperm(num_person) ;
    trainData = [trainData personData(:, 1:32)] ;
    trainLabels = [trainLabels person*ones(1, 32)] ;
    testData = [testData personData(:,33:end)] ;
    testLabels = [testLabels person*ones(1,num_person - 32)] ;
end
numClasses = size(unique(trainLabels),2) ;
inputSize = size(trainData, 1) ;
n_trainData = size(trainData, 2) ;

%% 1. initialize parameter
%%%% network
layersizes = [inputSize 1000 1000] ;
sparsityParamSet = 0.1;  %% 0.1 
lambdaSet = 0;%3e-3 ;            
betaSet = 0.3;%0.3 ;  % 1 0.1 0.01 0.001  
sparsityParam = sparsityParamSet ;
lambda = lambdaSet ;
beta = betaSet ;
%%%% optimization
options.Method = 'cg'; 
options.maxIter = 400 ;	
options.display = 'on';
%%%% classifier
c_lambda = 1e-4 ;
ft_lambda = c_lambda ;
%%%% usage
global useGpu ;
global tiedWeight;

useGpu = true ;
tiedWeight = true ;

%%%% cross validation
svNumSet = size(trainData, 2) ;
svNum = svNumSet ;
guassBeta = 0.5 ;
%%%% result 
acc = 0 ;
acc_FT = 0 ;
k_smooth = 3 ;


coef = zeros(n_trainData, k_smooth) ;
idxSm = zeros(n_trainData, k_smooth) ;
for cc = 1:numClasses
    index = find(trainLabels == cc) ;
    [knnIdx, dist] = knnsearch(trainData(:,index)', trainData(:,index)', 'k', k_smooth) ;
    idxSm(index,:) = index(knnIdx) ;
    coef(index,:) = bsxfun(@rdivide, exp(-dist*guassBeta), sum(exp(-dist*guassBeta),2));
end

%% 2. layerwise pretrain
input = trainData ;
theta = [] ;
for num_layer = 1:(length(layersizes)-1)
    saeTheta{num_layer} = initializeParameters(layersizes(num_layer+1), layersizes(num_layer));

    smData = sum(reshape(bsxfun(@times, input(:, idxSm(:)), coef(:)'),layersizes(num_layer),...
             n_trainData, k_smooth),3) ;
    [saeOptTheta{num_layer}, cost] = minFunc( @(p) smoothsparseTiedAECrossEntropyCost(p, layersizes(num_layer),...
                                     layersizes(num_layer+1), lambda, sparsityParam, beta, input, smData), ...
                                     saeTheta{num_layer}, options);

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
acc_FT = mean(testLabels(:) == pred_FT(:))
