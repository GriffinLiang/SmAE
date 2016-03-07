function [pred, output] = stackedAEPredict(theta, layersizes, numClasses, data)
                                         

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

depth = numel(stack) ;
a{1} = data ;
for layer = 1:depth
    z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
    a{layer+1} = sigmoid(z{layer+1}) ;
end

output = a{depth+1} ;
M = softmaxTheta * a{depth+1};
M = bsxfun(@minus, M, max(M));
p = bsxfun(@rdivide, exp(M), sum(exp(M)));

[~, pred] = max(p) ;

% -----------------------------------------------------------

end