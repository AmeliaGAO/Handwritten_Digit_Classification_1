function [y] = logistic_predict(weights, data)
%    Compute the probabilities predicted by the logistic classifier.
%
%    Note: N is the number of examples and 
%          M is the number of features per example.
%
%    Inputs:
%        weights:    (M+1) x 1 vector of weights, where the last element
%                    corresponds to the bias (intercepts).
%        data:       N x M data matrix where each row corresponds 
%                    to one data point.
%    Outputs:
%        y:          :N x 1 vector of probabilities. This is the output of the classifier.


N = size(data,1);
M = size(data,2);
z=zeros(N,1);
y=zeros(N,1);

for n=1:N
    for i = 1:M
        z(n,1)= z(n,1)+data(n,i)*weights(i,1);
    end
    z(n,1)=z(n,1)+weights(M+1,1);
    y(n,1) = 1/(1+exp(-z(n,1)));
end

end
