function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)

global tiedWeight;

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
if tiedWeight
    b1 = theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);
else
    b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
end
%-------------------------------------------------------------------
activation = sigmoid(bsxfun(@plus, W1*data, b1)) ;

end