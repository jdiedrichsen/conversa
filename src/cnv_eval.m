function varargout = cnv_eval(predictor, labels, algoNames, varargin)
% Evaluates learning algorithms

% Initialize optional arguments default values
optionArgs = struct( ... % TODO: Setup optionArgs with default vals and then set via getArgs
	'trainingsize', '0.8' ... % Testing with 80% of the data by default, giving 20% of the data for testing
	    );
%   'field', 'defaultval', ...

% Get and set args as provided
optionArgs = cnv_getArgs(optionArgs, varargin);

% Check optionArgs if needed

% Set lists of learning and 
learnPrefix = 'cnv_learn_';
predictionPrefix = 'cnv_predict_';
% Learning functions
learnFunctions = preSufFuncList({learnPrefix}, algoNames)';
predictFunctions = preSufFuncList({predictionPrefix}, algoNames)';

% Partition into learning set and testing set, cycle partitions and update
% error
trainingIndices = [];
testingIndices = [];

end

% Returns a function handle for the function with the name prefix ||
% suffix, i.e. the prefix and suffix concatenated as in @prefixsuffix
function out = preSufFunc(prefix, suffix)
out = str2func(strcat(prefix, suffix));
end

% Returns function handle list with all combos from the prefixList and
% suffixList
function out = preSufFuncList(prefixes, suffixes)
nPrefixes = length(prefixes);
nSuffixes = length(suffixes);
out = cell(nPrefixes, nSuffixes);
for i = 1:nPrefixes
    for j = 1:nSuffixes
        out{i, j} = preSufFunc(prefixes{i}, suffixes{j});
    end;
end;
end