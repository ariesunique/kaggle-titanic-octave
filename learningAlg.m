%% Machine Learning Online Class
% This is practice.
% Practice doing some of the earlier exercises using the optimization algorithm
% (fminunc) that we learned in week 4

function theta = learningAlg(X, y)

%% Load Data
%data = load('ex1data1.txt');
%X = data(:, 1:2);
%y = data(:, 3);
%[m, n] = size(X);

% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X);

% Add intercept term to x and X_test
%X = [ones(size(X, 1), 1) X];

options = optimset("GradObj", "on", "MaxIter", "400");
initialTheta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
%[theta, cost] = fminunc(@(t)(costFunctionLinear(t, X, y)), initialTheta, options);
[theta, cost] = fminunc(@(t)(costFunctionLogisticReg(t, X, y, lambda)), initialTheta, options);
%[theta, cost] = fminunc(@(t)(costFunctionLogistic(t, X, y)), initialTheta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
%fprintf('theta: \n');
%fprintf(' %f \n', theta);

end