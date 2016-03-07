function [cost,grad] = smoothsparseTiedAECrossEntropyCost(theta, data, smData, ae)

visibleSize = ae.visibleSize ;
hiddenSize = ae.hiddenSize ;
lambda = ae.lambda ;
sparsityParam = ae.sparsityParam ;
beta = ae.beta ;

global useGpu ;

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);
b2 = theta(hiddenSize*visibleSize+hiddenSize+1:end);
W1grad = zeros(size(W1)); 
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

if( useGpu == true)
    data = gpuArray( data );
end


n_data = size(data, 2) ;
hidden = sigmoid(W1*data+repmat(b1, 1, n_data)) ;
output = sigmoid(W1'*hidden+repmat(b2, 1, n_data)) ;

ave_hidden = sum(hidden, 2)/n_data ;
cost_sparse = sparsityParam*log(sparsityParam./ave_hidden)+(1-sparsityParam)*...
              log((1-sparsityParam)./(1-ave_hidden));

cost_smooth = -sum(sum(smData.*log(output)+(1-smData).*log(1-output)))/n_data;
          

cost = cost_smooth + beta*sum(cost_sparse)+lambda*sum(sum(W1.^2)); 
e_output = -(smData./output-(1-smData)./(1-output)).*output.*(1-output) ;
e_sparse = beta*(-sparsityParam./ave_hidden + (1-sparsityParam)./(1-ave_hidden)) ;
e_W2 = (W1*e_output+repmat(e_sparse,1,n_data)).*hidden.*(1-hidden) ;

W1grad = W1grad + e_W2*data'/n_data + lambda*W1 ;
W1grad = (W1grad' + e_output*hidden'/n_data + lambda*W1')' ;
b1grad = b1grad + sum(e_W2, 2)/n_data ;
b2grad = b2grad + sum(e_output, 2)/n_data ;

grad = [W1grad(:) ; b1grad(:) ; b2grad(:)];

if( useGpu == true)
    cost = gather(cost) ;
    grad = gather(grad) ;
end

end