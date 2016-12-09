%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
%load mnist_train_small;
load mnist_valid;
%load mnist_test;

%% Initialize hyperparameters.
lumbda = [0.001;0.01;0.1;1.0];
learning_rate = [0.1;0.1;0.1;0.1];
num_iterations = [500;500;500;500];
initial_weights = ['0.1*randn';'0.1*randn';'0.1*randn';'0.1*randn'];
num_rerun = 10;
s= size(lumbda,1);
n = size(train_inputs, 2);
avg_cross_entropy_valid = zeros(s,1);
avg_frac_correct_valid = zeros(s,1);

%% Start loop for lumbda
for p = 1:s
    hyperparameters.weight_regularization = lumbda(p,1);
    hyperparameters.learning_rate = learning_rate(p,1);
    hyperparameters.num_iterations = num_iterations(p,1);
    %argstr = [initial_weights(p,1), '(',n+1,',1)']; 
    weights = 0.1*randn(n+1,1);
    cross_entropy_valid_record = zeros(num_rerun,1);
    frac_correct_valid_record = zeros(num_rerun,1);
    
    % Inner loop for number of re-run
    for q = 1: num_rerun
        [cross_entropy_valid, frac_correct_valid]= run_logistric_regression(hyperparameters,weights,train_inputs, train_targets, valid_inputs, valid_targets);
        cross_entropy_valid_record(q,1) = cross_entropy_valid;
        frac_correct_valid_record(q,1) = frac_correct_valid;
    end
    avg_cross_entropy_valid(p,1) = mean(cross_entropy_valid_record);
    avg_frac_correct_valid(p,1) = mean(frac_correct_valid_record);
end

%% Plot out figure
figure; hold on
plot(lumbda, avg_cross_entropy_valid,'r');
plot(lumbda, (1-avg_frac_correct_valid)*100,'b');
xlabel('lumbda');
ylabel('cross entropy / classification error');