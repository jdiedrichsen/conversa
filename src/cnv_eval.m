function outError = cnv_eval(ld, algoNames, behaviourNames, varargin)
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
	'trainsize', 0.8, ... % Testing with 80% of the data by default, giving 20% of the data for testing
	'errorfunc', 'immse' ... % Defaults to mean square error
	    );
optionArgs = cnv_getArgs(optionArgs, varargin); % Get and set args as provided
% TODO: Check optionArgs for error (e.g. trainsize <= 0 or trainsize > 1)

% TODO: Refactor to work with struct instead of translating to matrix

% List of tracking fields
trackingFields = {'neckposx', 'neckposy', 'neckposz', 'neckrotx', 'neckroty', 'neckrotz', 'headposx', 'headposy', 'headposz', 'headrotx', 'headroty', 'headrotz', 'brow_up_l', 'brow_up_r', 'brow_down_l', 'brow_down_r', 'eye_closed_l', 'eye_closed_r', 'cheek_puffed_l', 'cheek_puffed_r', 'lips_pucker', 'lips_stretch_l', 'lips_stretch_r', 'lip_lower_down_l', 'lip_lower_down_r', 'smile_l', 'smile_r', 'frown_l', 'frown_r', 'jaw_l', 'jaw_r', 'jaw_open'};
nTrackingFields = length(trackingFields);
% Set predictor fields
nSamples = length(ld.timestamp);
predictors = zeros(nSamples, nTrackingFields);
for fieldNo = 1:nTrackingFields
	predictors(:,fieldNo) = ld.(trackingFields{fieldNo});
end;
% Set label fields
nBehaviours = length(behaviourNames);
labels = zeros(nSamples, nBehaviours);
for behavNo = 1:nBehaviours
	labels(:,behavNo) = ld.(behaviourNames{behavNo});
end;

% Set basic info about the data
nSamples = size(predictors, 1);
if (nSamples ~= size(labels, 1)) % Labels and predictors must have corresponding rows
	error('The number of rows in the predictor matrix and label mamtrix are not equal');
end;
nTrainSamples = round(nSamples*(optionArgs.trainsize)); % Number of rows, since 
nTestSamples = nSamples - nTrainSamples; % All non-training samples are for testing
nPartitions = ceil((1)/(1-optionArgs.trainsize));

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
	predictFunc(algoName) = preSufFunc(predictPrefix, algoName);
end;

% Function signatures for learning, prediction:
%	model = cnv_learn_algo(predictors, labels)
%	predictedLabels = cnv_predict_algo(model, predictors)

% Train and test each algorithm
evalError = zeros(nAlgos, nPartitions+1); % Error matrix will have error of each algorithm (row), partition (column), and average error of algorithm (final column)
for algoNo = 1:nAlgos
	algoName = algoNames{algoNo};
	learn = learnFunc(algoName);
	predict = predictFunc(algoName);
	for partitionNo = 1:nPartitions
		% Train from 1 to testStartI-1 and testEndI+1 to nSamples
		testStart = testStartI(partitionNo);
		testEnd = testEndI(partitionNo);
		rows = [1:(testStart-1) (testEnd+1):nSamples];
		predictorSet = predictors(rows,:);
		labelSet = labels(rows,:);
		model = learn(predictorSet, labelSet);
		% Make prediction, evaluate, and update error
		evalError(algoNo, partitionNo) = findError( ...
			predict(model, predictors(max(testStart,1):min(testEnd,nSamples),:)), ... % Predicted by model
			labels(max(testStart,1):min(testEnd,nSamples)), ... % Actual labels
			optionArgs.errorfunc);
	end;
	evalError(algoNo, nPartitions+1) = mean(evalError(algoNo, 1:nPartitions)); % Update average error
end;

outError = evalError;

end % cnv_eval

% Returns a function handle for the function with the name prefix ||
% suffix, i.e. the prefix and suffix concatenated as in @prefixsuffix
function out = preSufFunc(prefix, suffix)
out = str2func(strcat(prefix, suffix));
end

function out = findError(predicted, actual, errorFuncStr)
errorFunc = str2func(errorFuncStr);
out = errorFunc(predicted, actual);
end