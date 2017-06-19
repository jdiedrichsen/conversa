function varargout = cnv_eval(predictors, labels, algoNames, varargin)
% Evaluates learning algorithms
% Takes a matrix of predictors, array of labels, and cell array of the
% names of the algorithms to evaluate
% Returns an array of error rates for each algorithm in algoNames
% By Shayaan Syed Ali
% Last updated 19-Jun-17

% SET PARAMETERS ==========================================================
% Process varargin to apply input arguments as needed

% Initialize optional arguments default values
% Format is struct('fieldName1', 'defaultValue1', 'fieldName2', 'defaultValue2', ...)
optionArgs = struct( ... % TODO: Setup optionArgs with default vals and then set via getArgs
	'trainsize', '0.8' ... % Testing with 80% of the data by default, giving 20% of the data for testing
	    );
optionArgs = cnv_getArgs(optionArgs, varargin); % Get and set args as provided
% TODO: Check optionArgs for error (e.g. trainsize <= 0 or trainsize > 1)

% Set basic info about the data
nSamples = size(predictors, 1);
if (nSamples ~= size(labels, 1)) % Labels and predictors must have corresponding rows
	error('The number of rows in the predictor matrix and label mamtrix are not equal');
end;
nTrainSamples = round(nSamples*(optionArgs.trainsize)); % Number of rows, since 
nTestSamples = nSamples - nTrainSamples; % All non-training samples are for testing
nPartitions = ceil(1/optionArgs.trainsize);

% PARTITION DATA ==========================================================
% Partition data into learning and testing sets for training and evaluation

testStartI = zeros(nPartitions, 1);
testEndI = zeros(nPartitions, 1);
% Set test partition indices except last partition
for i = 1:(nPartitions-1)
	testStartI(i) = (i-1)*nTestSamples;
	testEndI(i) = (i)*nTestSamples;
end;
% Set last partition, may overlap with second last partition
testStartI(nPartitions) = nSamples - nTestSamples;
testEndI(nPartitions) = nSamples;

% TRAIN AND TEST ==========================================================
% Train and test algorithms, returning an error rate

% Set map from the algoNames to the learning and prediction functions
learnPrefix = 'cnv_learn_';
predictPrefix = 'cnv_predict_';
learnFunc = containers.Map();
predictFunc = containers.Map();
nAlgos = length(algoNames);
for i = 1:nAlgos
	algoName = algoNames{i};
	learnFunc(algoName) = preSufFunc(learnPrefix, algoName);
	predictFunc(algoName)  = preSufFunc(predictPrefix, algoName);
end;

% Function signatures for learning, prediction:
%	model = cnv_learn_algo(predictors, labels)
%	predictedLabels = cnv_predict_algo(model, predictors)

% Train and test each algorithm
error = zeroes(algoNo, nPartitions+1); % Error matrix will have error of each algorithm (row), partition (column), and average error of algorithm (final column)
for algoNo = 1:nAlgos
	algoName = algoNames{algoNo};
	learn = learnFunc(algoName);
	predict = predictFunc(algoName);
	for partitionNo = 1:nPartitions
		% Train from 1 to testStartI-1 and testEndI+1 to nSamples
		firstSetEnd = testStartI(partitionNo) - 1;
		secondSetStart = testEndI(partitionNo) + 1;
		predictorSet = predictors([1:firstSetEnd secondSetStart:nSamples],:);
		labelSet = labels([1:firstSetEnd secondSetStart:nSamples],:);
		learn(predictorSet, labelSet);
		% Test from testStartI to testEndI and update error
		
	end;
end;

end % cnv_eval

% Returns a function handle for the function with the name prefix ||
% suffix, i.e. the prefix and suffix concatenated as in @prefixsuffix
function out = preSufFunc(prefix, suffix)
out = str2func(strcat(prefix, suffix));
end