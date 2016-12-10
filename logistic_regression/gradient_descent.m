function [f, df, y] = gradient_descent(weights, data, targets, hyperparameters)
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%	targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value.
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%

N = size(data,1);
M = size(data,2);
if(size(weights,1) ~= M+1)
    error('Weight size is not equal to number of columns of data plus 1');
end

if(size(targets,1) ~= N)
    error('targets size is not equal to number of rows in data');
end

y = logistic_predict(weights, data);
[f, frac_correct] = evaluate(targets, y);

dw = data'*(y-targets);
db = sum(y-targets);
df = [dw;db];

end
