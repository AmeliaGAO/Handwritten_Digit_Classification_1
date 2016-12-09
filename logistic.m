function [f, df, y] = logistic(weights, data, targets, hyperparameters)
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

f=0;
df = zeros(M+1,1);
y = zeros(N,1);
z = zeros(N,1);

for n=1:N
    for d = 1:M
        z(n,1)= z(n,1)+data(n,d)*weights(d,1);
    end
    z(n,1)=z(n,1)+weights(M+1,1);
    f=f+log(1+exp(-z(n,1))) + (targets(n,1)-1)*(-z(n,1));
    y(n,1) = 1/(1+exp(-z(n,1)));
end

for d = 1:M
    for n=1:N
        df(d,1) = df(d,1)-data(n,d)*(exp(-z(n,1))/(1+exp(-z(n,1)))+targets(n,1)-1);
    end
end

for n=1:N
    df(M+1,1) = df(M+1,1)-(exp(-z(n,1))/(1+exp(-z(n,1)))+targets(n,1)-1);
end

end
