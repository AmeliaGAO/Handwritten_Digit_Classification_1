function r = cal_classification_rate(results, targets)
%    Compute classfication rate.
%    Inputs:
%        targets : N x 1 vector of targetes.
%        results : N x 1 vector of classification result.
%    Outputs:
%        r : number of correctly predicted cases, divided by total number
%        of data points.

error(nargchk(2,2,nargin));

if (size(results,1) ~= size(targets,1))
   error('Results and Targets should be of same dimensionality');
end

N = size(results,1);
r = sum(results==targets)/N;
end