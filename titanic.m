clear ;

%% Load Data
data = load("train_modified.csv");
X = data(:, 2:7);
y = data(:, 1);
clear data;

% Print out some data points
%fprintf('First 10 examples from the dataset: \n');
%fprintf(' x = [%.0f %.0f %.0f ], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

theta = learningAlg(X, y);

X_guess = load("test_modified.csv");
%X_guess = data(501:700, 2:7);
X_guess = mapFeature(X_guess);
%X_guess = [ones(size(X_guess, 1), 1) X_guess];
h = sigmoid(X_guess * theta);

prediction = zeros(length(h),1);
prediction(find(h>=0.5)) = 1;

%actual = data(501:700, 1);
%fprintf('Train Accuracy: %f\n', mean(double(prediction == actual)) * 100 );

fprintf("Saving prediction to prediction.txt \n");
save prediction.txt prediction;