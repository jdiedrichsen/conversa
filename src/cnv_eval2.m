function out = cnv_eval2(data, algoNames, targetFields, varargin)
% Evaluates learning algorithms
% 
% Takes a data struct with labelled training data, a cell array of the
% names of the algorithms to be tested, and a few optional argument
% parameters
% 
% Assumes that each algorithm name in algoNames is accompanied by a
% Matlab script with the prefix cnv_learn_algoNames{i} and
% cnv_predict_algoNames{i} for learning and prediction
% (classification/regression) respectively
% Also assumes that all fields in data not in targetFields are for training
% unless otherwise specified in the optional parameter excludefields
% 
% Returns a struct of error rates for each algorithm in algoNames
% 
% By Shayaan Syed Ali
% Last updated 27-Jun-17

% Set parameters (optional arguments, and structs of predictors and labels)

optionArgs = struct( ...
	'trainratio', 0.8, ... % The ratio of the data to show algorithms before testing, 80% by default
	'npartitions', 5, ... % The number of partitions to break the data into
	'errorfunc', 'immse', ... % The error function to use for evaluating models after training
	'verbose', false, ...
	'excludefields', 'none' ...
	);
optionArgs = cnv_getArgs(optionArgs, varargin);
if (optionArgs.verbose), disp('cnv_eval: Initialized optional arguments, setting predictor and label structs'), end;
% Setting up structs of predictors and labels
if (~strcmp(optionArgs.excludefields, 'none')) % Strip away excludefields if required
	if (optionArgs.verbose), disp('cnv_eval: Removing fields: '), fprintf('\b'), disp(optionArgs.excludefields), end;
	data = rmfield(data, optionArgs.excludefields);
end
predictors = rmfield(data, targetFields);
labels = rmfield(data, fieldnames(predictors)); % Everything which is not a predictor is a label
if (optionArgs.verbose), disp('cnv_eval: Predictor and label structs set, partitioning data'), end;

% Partition data


% Train and test


end % cnv_eval2