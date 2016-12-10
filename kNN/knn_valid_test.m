%% Clear workspace.
clear all;
close all;

%% Load data.
load('mnist_train.mat');
load('mnist_valid.mat');
load('mnist_test.mat');

%% Initiate k values
k_values = [1,3,5,7,9];

%% Plot classification_rate - k_value figure for validation set.
plot_k_cr(k_values, train_inputs, train_targets, valid_inputs, valid_targets);
title('Classification Rate for Validation Set');

%% Plot classification_rate - k_value figure for test set.
figure();
plot_k_cr(k_values, train_inputs, train_targets, test_inputs, test_targets);
title('Classification Rate for Test Set');
