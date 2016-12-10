%% Clear workspace.
clear all;
close all;

%% Load data.
%load mnist_train;
load mnist_train_small;
load mnist_valid;
load mnist_test;


%% Initialize hyperparameters.
hyperparameters.learning_rate = 0.1;
hyperparameters.weight_regularization = 0.1;
hyperparameters.num_iterations = 500;
% Logistics regression weights
% Set random weights.
n = size(train_inputs_small, 2);
weights = 0.1*randn(n+1,1);

%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('gradient_descent', ... 
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                   % other hyperparameters
N = size(train_inputs_small, 1);

%% Begin learning with gradient descent.
best_frac_correct_valid = 0;
best_weights = zeros(n+1,1);
train_cross_entropy_record = zeros(hyperparameters.num_iterations,1);
train_correct_rate_record = zeros(hyperparameters.num_iterations,1);
valid_cross_entropy_record = zeros(hyperparameters.num_iterations,1);
valid_correct_rate_record = zeros(hyperparameters.num_iterations,1);

for t = 1:hyperparameters.num_iterations

	% Find the negative log likelihood and derivative w.r.t. weights.
	[f, df, predictions] = gradient_descent(weights, ...
                                    train_inputs_small, ...
                                    train_targets_small, ...
                                    hyperparameters);

    [cross_entropy_train, frac_correct_train] = evaluate(train_targets_small, predictions);

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
        best_weights = weights;
    end
        
	%% Print some stats.
	fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
			t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);
        
    train_cross_entropy_record(t,1) = cross_entropy_train;
    train_correct_rate_record(t,1) = frac_correct_train*100;
    valid_cross_entropy_record(t,1) = cross_entropy_valid;
    valid_correct_rate_record(t,1) = frac_correct_valid*100;
end

%% Plot and display
x=[1:hyperparameters.num_iterations]';
figure; hold on
a1=plot(x, train_cross_entropy_record,'r--'); 
a2=plot(x, train_correct_rate_record,'r'); 
a3=plot(x, valid_cross_entropy_record,'b--'); 
%xlabel('iteration');
%ylabel('cross entropy');
a4=plot(x, valid_correct_rate_record,'b'); 
%legend([a1,a2,a3,a4],[M1,M2,M3,M4]);
%plot(x,train_cross_entropy_record,'r--',x, train_correct_rate_record,'r', x, valid_cross_entropy_record,'b--',x,valid_correct_rate_record,'b');
predictions_test = logistic_predict(best_weights,test_inputs);

display(best_frac_correct_valid);
[cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test)
