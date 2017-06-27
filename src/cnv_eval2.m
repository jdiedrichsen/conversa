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
	'trainratio', 0.8, ... % The ratio of the data to show algorithms before testing, 80% by default, implying 20% for testing
	'nparts', 5, ... % The number of partitions to break the data into for training and testing
	'partitiontype', 'random', ... % The partition type to make, can set to random, 
	'errorfunc', 'immse', ... % The error function to use for evaluating models after training
	'ntests', 1, ... % The number of overall times to test the set of partitions, for example nparts = 5 and ntests = 2 would train and test the models 10 times
	'verbose', false, ...
	'excludefields', 'none' ...
	);
optionArgs = cnv_getArgs(optionArgs, varargin);
v = optionArgs.verbose; % For brevity
if (v), disp('cnv_eval: Initialized optional arguments, setting predictor and label structs'); end
% Setting up structs of predictors and labels
if (~strcmp(optionArgs.excludefields, 'none')) % Strip away excludefields if required
	if (v), disp('cnv_eval: Removing fields: '); fprintf('\b'); disp(optionArgs.excludefields); end
	data = rmfield(data, optionArgs.excludefields);
end
predictors = rmfield(data, targetFields);
labels = rmfield(data, fieldnames(predictors)); % Everything which is not a predictor is a label
if (v), disp('cnv_eval: Predictor and label structs set, partitioning data'); end
predictorFields = fieldnames(predictors);
labelFields = fieldnames(labels);
% Set some basic info to be used when partitioning
nSamples = length(predictors.(predictorFields{1}));
if (nSamples ~= length(labels.(labelFields{1}))) % Can make this error call more robust and check all fields
	error('Number of predictors and labels is not equal');
end
nTestSamples = nSamples*(1-optionArgs.trainratio);

% Partition data

partitions = zeros(optionArgs.nparts, 2); % Matrix of starts and ends for partitions
switch (optionArgs.partitiontype)
	case 'random'
		if (v), disp('cnv_eval: Setting random partitions'); end
		for i = 1:optionArgs.nparts
			partitions(i,1) = 1 + round((nSamples-nTestSamples)*rand);
			partitions(i,2) = partitions(i,1) + nTestSamples - 1;
		end
	case 'contiguous' % TO DO: test contiguous partitioning
		if (v), disp('cnv_eval: Setting contiguous partitions'); end
		if (optionArgs.nparts*nTestSamples > nSamples)
			error('To many partitions for contiguous partitioning');
		end
		for i = 1:optionArgs.nparts
			partitions(i,1) = 1 + (i-1)*nTestSamples;
			partitions(i,2) = partitions(i,1) + nTestSamples - 1;
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

for i=1:optionArgs.nparts
	
end

end % cnv_eval2