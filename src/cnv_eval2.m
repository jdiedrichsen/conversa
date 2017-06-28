function outError = cnv_eval2(data, algoNames, targetFields, varargin)
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
nParts = optionArgs.nparts;
nTests = optionArgs.ntests;
if (v), disp('cnv_eval: Initialized optional arguments'); end
% Setting up structs of predictors and labels
if (v), disp('cnv_eval: Setting predictor and label structs'); end
if (~strcmp(optionArgs.excludefields, 'none')) % Strip away excludefields if required
	if (v), disp('cnv_eval: Removing fields: '); fprintf('\b'); disp(optionArgs.excludefields); end
	data = rmfield(data, optionArgs.excludefields);
end
predictors = rmfield(data, targetFields);
labels = rmfield(data, fieldnames(predictors)); % Everything which is not a predictor is a label
if (v), disp('cnv_eval: Predictor and label structs set successfully'); end

% Partition data

% Set some basic info to be used when partitioning
predictorFields = fieldnames(predictors);
labelFields = fieldnames(labels);
nSamples = length(predictors.(predictorFields{1}));
if (nSamples ~= length(labels.(labelFields{1}))) % ADD: Can make this error call more robust and check all field lengths
	error('Number of predictors and labels is not equal');
end
nTestSamples = nSamples*(1-optionArgs.trainratio);

% Partition using selected method
testParts = zeros(nParts, 2); % Each row is a partition
switch (optionArgs.partitiontype)
	case 'random'
		if (v), disp('cnv_eval: Setting random partitions'); end
		for i = 1:nParts
			testParts(i,1) = 1 + round((nSamples-nTestSamples)*rand);
			testParts(i,2) = testParts(i,1) + nTestSamples - 1;
		end
	case 'contiguous' % TO DO: test contiguous partitioning
		if (v), disp('cnv_eval: Setting contiguous partitions'); end
		if (nParts*nTestSamples > nSamples)
			error('To many partitions for contiguous partitioning');
		end
		% ADD: Can be refactored to have sliding window behaviour when
		% partitions cannot give mutually exclusive testing sets
		for i = 1:nParts
			testParts(i,1) = 1 + (i-1)*nTestSamples;
			testParts(i,2) = testParts(i,1) + nTestSamples - 1;
		end
	otherwise
		if (v), disp('cnv_eval: Partition type not found'); end
		error('Invalid partitiontype');
end
if (v), disp('cnv_eval: Data partitioned successfully'); end

% Train and test

% Function signatures for learning, prediction:
%	model = cnv_learn_algo(predictors, labels)
%	predictedLabels = cnv_predict_algo(model, predictors)

% Set learning and prediction function map
learnFunc = containers.Map();
predictFunc = containers.Map();
nAlgos = length(algoNames);
for i = 1:nAlgos
	algoName = algoNames{i};
	lF = preSufFunc(LEARN_PREFIX, algoName);
	if (~exist(lF, 'file'))
		error('Learning function not found');
	end
	learnFunc(algoName) = lF;
	pF = preSufFunc(PREDICT_PREFIX, algoName);
	if (~exist(pF, 'file'))
		error('Prediction function not found');
	end
	predictFunc(algoName) = pF;
end
% Iterate through partitions, train and test, update error
outError = struct();
models = cell(nTests, nParts, nAlgos);
predictions = cell(nTests, nTestSamples, nAlgos);
for i = 1:nTests
	for j = 1:nParts
		testPart = testParts(j, 1):testParts(j, 2);
		trainPart = setdiff(1:nSamples, testPart);
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
		% Train and get models
		for k = 1:nAlgos
			algoName = algoNames{k};
			learn = learnFunc(algoName);
			models{i,j,k} = learn(predictTrainPart, labelTrainPart);
		end
		% Get model predictions
		for k = 1:nAlgos
			algoName = algoNames{k};
			predict = predictFunc(algoName);
			predictions{i,:,k} = predict(models{i,j,k}, predictTestPart);
		end
		% Update error
		for k = 1:nAlgos
			algoName = algoNames{k};
			outError.(algoName)((i-1)*nParts + j) = evalError(predictions{i,:,k}, labelTestPart, optionArgs.errorfunc);
		end
	end
end

end % cnv_eval2

% Returns a function handle for the function with the name prefix ||
% suffix, i.e. the prefix and suffix concatenated as in @prefixsuffix
function out = preSufFunc(prefix, suffix)
out = str2func(strcat(prefix, suffix));
end

function out = evalError(predicted, actual, errorFuncStr)
errorFunc = str2func(errorFuncStr);
out = errorFunc(predicted, actual);
end