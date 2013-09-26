function [J, grad] = costFunctionLogisticReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of parameters

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


[J, grad] = costFunctionLogistic(theta, X, y);
theta1 = theta(1);
J = J + (lambda/(2*m))*sum(theta.^2) - (lambda/(2*m))*theta1^2;

for i = 2:n
	grad(i) = grad(i) + (lambda/m) * theta(i);
end


% =============================================================

end
