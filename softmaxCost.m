function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

M = theta*data ;
M = bsxfun(@minus, M, max(M, [], 1)) ;
J_cost = -sum(log(sum(groundTruth.*exp(M))./sum(exp(M))))/numCases ;
J_Weight = 0.5*lambda*sum(sum(theta.^2)) ;
cost = J_cost + J_Weight ;
P = bsxfun(@rdivide, exp(theta*data), sum(exp(theta*data))) ;
thetagrad = -(groundTruth-P)*data'/numCases + lambda*theta ;
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

% function y = sigmoid(x)
%     y = 1./ (1 + exp(-x)) ;
% end