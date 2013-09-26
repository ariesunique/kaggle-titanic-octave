function out = mapFeature(X)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%
	
degree = 2;
out = ones(size(X));
for i = 1:degree
    for j = 0:i
		for k = 0:j
			for m = 0:k
				for n = 0:m
					for o = 0:n
						out(:, end+1) = (X(:,1).^(i-j)).*(X(:,2).^(j-k)).*(X(:,3).^(k-m)).*(X(:,4).^(m-n)).*(X(:,5).^(n-o)).*(X(:,6).^o);
					end
				end
			end
		end
    end
end

end