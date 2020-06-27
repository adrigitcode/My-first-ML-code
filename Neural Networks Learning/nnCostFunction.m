function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Feedforward and cost function
A1= [ones(m,1), X];
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);
A2 = [ones(m,1), A2];
Z3 = A2 * Theta2';
ho = sigmoid(Z3);
 
yk = zeros(m,num_labels);
 
for i = 1:m
    j = y(i);
    yk(i, j) = 1;
end
 
E = sum((yk.*log(ho)) + ((1-yk).*log(1-ho)));
J = sum((-1/m)*E);
        
%Regularized cost function
O1w = Theta1(:,2:end);
O2w = Theta2(:,2:end);
 
E1 = sum(sum(O1w.^2));
E2 = sum(sum(O2w.^2));
 
reg = lambda/(2*m)*(E1+E2);
J = J + reg;

%Backpropagation
D_1 = zeros(size(Theta1)); %Accumulated gradient for layer 2
D_2 = zeros(size(Theta2)); %Accumulated gradient for layer 3
 
for t = 1:m
    delta3 = ho(t,:)' - yk(t,:)';
    delta2 = (Theta2' * delta3).*(sigmoidGradient([1,Z2(t,:)]))';
    delta2 = delta2(2:end);
             
    D_1 = D_1 + delta2 * A1(t,:);
    D_2 = D_2 + delta3 * A2(t,:);
end

Theta1_grad =  (1/m) * D_1; %unregularized gradient for layer 2
Theta2_grad =  (1/m) * D_2; %unregularized gradient for layer 3

Theta1_grad(:,2:end) = ((1/m) * D_1(:,2:end)) + ((lambda/m) * Theta1(:,2:end)); %regularized gradient for layer 2
Theta2_grad(:,2:end) = ((1/m) * D_2(:,2:end)) + ((lambda/m) * Theta2(:,2:end)); %regularized gradient for layer 3 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
