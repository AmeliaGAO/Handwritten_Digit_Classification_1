function [cross_entropy_valid, best_frac_correct_valid] = run_logistic_regression(hyperparameters,weights,train_inputs, train_targets,valid_inputs, valid_targets)

best_frac_correct_valid = 0;
N = size(train_inputs, 1);
for t = 1:hyperparameters.num_iterations

	% Find the negative log likelihood and derivative w.r.t. weights.
	[f, df, predictions] = logistic_pen(weights, ...
                                    train_inputs, ...
                                    train_targets, ...
                                    hyperparameters);

    [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions);

	% Find the fraction of correctly classified validation examples.
	%[temp, temp2, frac_correct_valid] = logistic(weights, ...
    %                                             valid_inputs, ...
    %                                             valid_targets, ...
    %                                             hyperparameters);

    if isnan(f) || isinf(f)
		error('nan/inf error');
	end

    %% Update parameters.
    weights = weights - hyperparameters.learning_rate .* df / N;

    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    
    %% Find the best hyperparameters to get the highest frac_correct_valid
    if frac_correct_valid > best_frac_correct_valid
        best_frac_correct_valid = frac_correct_valid;
    end
        
	%% Print some stats.
	fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
			t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);
        
end

end