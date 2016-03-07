function [ cost, grad ] = FinetuneAECost(theta, layersizes,numClasses,lambda, data, labels)
                                         

global useGpu;

softmaxTheta = reshape(theta(1:layersizes(end)*numClasses), numClasses, layersizes(end));

stackAETheta = theta(layersizes(end)*numClasses+1:end);
l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    stack{i}.w = reshape(stackAETheta(lold:lnew), layersizes(i+1), layersizes(i));
    lold = lnew + 1;
    lnew = lnew + layersizes(i+1);
    stack{i}.b = stackAETheta(lold:lnew);
end

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

if useGpu
    data = gpuArray( data );
end

M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

s{1} = sigmoid(stack{1}.w*data+repmat(stack{1}.b, 1, M)) ;
for ii=2:l-1
    s{ii} = sigmoid(stack{ii}.w*s{ii-1}+repmat(stack{ii}.b, 1, M)) ;
end

M_out = softmaxTheta*s{l-1} ;
M_out = bsxfun(@minus, M_out, max(M_out, [], 1)) ;
J_cost = -sum(log(sum(groundTruth.*exp(M_out))./sum(exp(M_out))))/numClasses ;
J_Weight = 0.5*lambda*sum(softmaxTheta(:).^2) ;
cost = J_cost + J_Weight ;

P = bsxfun(@rdivide, exp(M_out), sum(exp(M_out))) ;
softmaxThetaGrad = -(groundTruth-P)*s{l-1}'/numClasses + lambda*softmaxTheta ;


e{l} = -softmaxTheta'*(groundTruth-P).*(s{l-1}).*(1-s{l-1}) ;

for ii=(l-1):-1:2
    stackgrad{ii}.w = e{ii+1}*s{ii-1}'/numClasses  ;
    stackgrad{ii}.b = sum(e{ii+1}, 2)/numClasses ;
    e{ii} = (stack{ii}.w'*e{ii+1}).*s{ii-1}.*(1-s{ii-1}) ;
end

stackgrad{1}.w = e{2}*data'/numClasses  ;
stackgrad{1}.b = sum(e{2}, 2)/numClasses ;

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

if useGpu
    cost = gather(cost) ;
    grad = gather(grad) ;
end

end