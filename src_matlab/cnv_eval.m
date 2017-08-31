function outError = cnv_eval(data, algoNames, targetFields, varargin)
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

% Constants

LEARN_PREFIX = 'cnv_learn_';
PREDICT_PREFIX = 'cnv_predict_';
ERROR_FIELD_SUFFIX = '_error';

% Set parameters (optional arguments, and structs of predictors and labels)

optionArgs = struct( ...
	'trainratio', 0.8, ... % The ratio of the data to show algorithms before testing, 80% by default, implying 20% for testing
	'nparts', 5, ... % The number of partitions to break the data into for training and testing
	'partitiontype', 'random', ... % The partition type to make, can set to random, 
	'errorfunc', 'immse', ... % The error function to use for evaluating models after training
	'ntests', 1, ... % The number of overall times to test the set of partitions, for example nparts = 5 and ntests = 2 would train and test the models 10 times
	'verbose', false, ...
	'excludefields', 'none' ...
	);
optionArgs = cnv_getArgs(optionArgs, varargin);
% For brevity
v = optionArgs.verbose;
if (v), fprintf('cnv_eval.m: Beginning execution\n'); end
nParts = optionArgs.nparts;
nTests = optionArgs.ntests;
if (v), fprintf('cnv_eval.m: Initialized optional arguments\n'); end
% Setting up structs of predictors and labels
if (v), fprintf('cnv_eval.m: Setting predictor and label structs\n'); end
if (~strcmp(optionArgs.excludefields, 'none')) % Strip away excludefields if required
	if (v), fprintf('cnv_eval.m: Removing fields: '); disp(optionArgs.excludefields); end
	data = rmfield(data, optionArgs.excludefields);
end
predictors = rmfield(data, targetFields);
labels = rmfield(data, fieldnames(predictors)); % Everything which is not a predictor is a label
if (v), fprintf('cnv_eval.m: Predictor and label structs set successfully\n'); end

% Partition data

% Set some basic info to be used when partitioning
predictorFields = fieldnames(predictors);
labelFields = fieldnames(labels);
nSamples = length(predictors.(predictorFields{1}));
if (nSamples ~= length(labels.(labelFields{1}))) % ADD: Can make this error call more robust and check all field lengths
	error('Number of predictors and labels is not equal');
end
nTestSamples = round(nSamples*(1-optionArgs.trainratio));

% Set parition indices using selected method
testParts = zeros(nParts, 2); % Each row is a partition
switch (optionArgs.partitiontype)
	case 'random'
		if (v), fprintf('cnv_eval.m: Setting random partition indices\n'); end
		for i = 1:nParts
			testParts(i,1) = 1 + round((nSamples-nTestSamples)*rand);
			testParts(i,2) = testParts(i,1) + nTestSamples - 1;
		end
		if (v), fprintf('cnv_eval.m: Random test partitions indices set at:\n'); disp(testParts); end
	case 'contiguous' % TO DO: test contiguous partitioning
		if (v), fprintf('cnv_eval.m: Setting contiguous partition indices\n'); end
		if (nParts*nTestSamples > nSamples)
			error('Too many partitions for contiguous partitioning');
		end
		% ADD: Can be refactored to have sliding window behaviour when
		% partitions cannot give mutually exclusive testing sets
		for i = 1:nParts
			testParts(i,1) = 1 + (i-1)*nTestSamples;
			testParts(i,2) = testParts(i,1) + nTestSamples - 1;
		end
	otherwise
		if (v), fprintf('cnv_eval.m: Partition type not found\n'); end
		error('Invalid partitiontype');
end
if (v), fprintf('cnv_eval.m: Data partition indices set successfully\n'); end

% Train and test

% Function signatures for learning, prediction:
%	model = cnv_learn_algo(predictors, labels)
%	predictedLabels = cnv_predict_algo(model, predictors)

% Set learning and prediction function map
if (v), fprintf('cnv_eval.m: Setting learning and prediction functions\n'); end
learnFunc = containers.Map();
predictFunc = containers.Map();
nAlgos = length(algoNames);
for i = 1:nAlgos
	algoName = algoNames{i};
	% Set learning function
	lfName = strcat(LEARN_PREFIX, algoName);
	if (~exist(lfName, 'file'))
		error('Learning function not found');
	end
	learnFunc(algoName) = str2func(lfName);
	% Set prediction function
	pfName = strcat(PREDICT_PREFIX, algoName);
	if (~exist(pfName, 'file'))
		error('Prediction function not found');
	end
	predictFunc(algoName) = str2func(pfName);
end
if (v), fprintf('cnv_eval.m: Learning and prediction functions set successfully\n'); end
% Iterate through partitions, train and test, update error
if (v), fprintf('cnv_eval.m: Beginning training and testing\n'); end
outError = struct();
models = cell(nTests, nParts, nAlgos);
predictions = cell(nTests, nAlgos);
for i = 1:nTests
	if (v), fprintf('cnv_eval.m: Test number: %d\n', i); end
	for j = 1:nParts
		if (v), fprintf('cnv_eval.m: Partition number: %d\n', j); end
		testPart = testParts(j,1):testParts(j,2);
		trainPart = setdiff(1:nSamples, testPart);
		if (v), fprintf('cnv_eval.m: Partitioning predictor and label data\n'); end
		% Partition predictors
		predictTestPart = struct();
		predictTrainPart = struct();
		for k = 1:length(predictorFields)
			field = predictorFields{k};
			predictTestPart.(field) = predictors.(field)(testPart);
			predictTrainPart.(field) = predictors.(field)(trainPart);
		end
		% Partition labels
		labelTestPart = struct();
		labelTrainPart = struct();
		for k = 1:length(labelFields)
			field = labelFields{k};
			labelTestPart.(field) = labels.(field)(testPart);
			labelTrainPart.(field) = labels.(field)(trainPart);
		end
		if (v), fprintf('cnv_eval.m: Partitioning complete\n'); end
		% Train and get models
		if (v), fprintf('cnv_eval.m: Beginning training\n'); end
		for k = 1:nAlgos
			algoName = algoNames{k};
			if (v), fprintf('cnv_eval.m: Training %s\n', algoName); end
			learn = learnFunc(algoName);
			models{i,j,k} = learn(predictTrainPart, labelTrainPart);
			if (v), fprintf('cnv_eval.m: %s trained\n', algoName); end
		end
		if (v), fprintf('cnv_eval.m: All training complete\n'); end
		% Get model predictions
		if (v), fprintf('cnv_eval.m: Getting predictions on test set\n'); end
		for k = 1:nAlgos
			algoName = algoNames{k};
			if (v), fprintf('cnv_eval.m: Predicting with %s\n', algoName); end
			predict = predictFunc(algoName);
			predictions{i,k} = predict(models{i,j,k}, predictTestPart);
			if (v), fprintf('cnv_eval.m: %s completed prediction\n', algoName); end
		end
		if (v), fprintf('cnv_eval.m: All predictions complete\n'); end
		% Update error
		if (v), fprintf('cnv_eval.m: Evaluating error with %s\n', optionArgs.errorfunc); end
		for k = 1:nAlgos
			algoName = algoNames{k};
			predictionError = evalError(predictions{i,k}, cnv_struct2Matrix(labelTestPart), optionArgs.errorfunc);
			if (v), fprintf('cnv_eval.m: %s had an error of %f\n', algoName, predictionError); end
			outError.(strcat(algoName, ERROR_FIELD_SUFFIX))((i-1)*nParts + j,1) = predictionError;
		end
	end
end
if(v), fprintf('cnv_eval.m: Completed execution\n'); end
end % cnv_eval

% Recieves predicted and actual as matrices (or vectors)
function out = evalError(predicted, actual, errorFuncStr)
errorFunc = str2func(errorFuncStr);
out = errorFunc(predicted, actual);
end