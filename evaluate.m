function [ce, frac_correct] = evaluate(targets, y)
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of binary targets. Values should be either 0 or 1.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we
%                       want to compute CE(targets, y).
%        frac_correct : (scalar) Fraction of inputs classified correctly.


N = size(targets,1);
ce=0;
nCorrect=0;
for n=1:N
    p = targets(n,1);
    q = y(n,1);
    ce = ce - p*(log(q)) - (1-p)*(log(1-q));
    if (q<0.5 && p==0)||(q>=0.5 && p==1)
        nCorrect = nCorrect +1;
    end
end
    
frac_correct = nCorrect/N;

end
