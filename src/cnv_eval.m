function varargout = cnv_eval(predictor, labels, algoNames, varargin)
% Evaluates learning algorithms

% Set optional arguments default values
optionArgs = struct( ... % TODO: Setup optionArgs with default vals and then set via getArgs
    'field', 'val' ...
    );
% Get and set args as provided


% Set lists of learning and 
learnPrefix = 'cnv_learn_';
predictionPrefix = 'cnv_predict_'; 
% Learning functions
learnFunctions = preSufFuncList({learnPrefix}, algoNames)
predictFunctions = preSufFuncList({predictionPrefix}, algoNames)

% Learn with a partition of the data


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
out = cell(nPrefixes, nSuffixes)
for i = 1:nPrefixes
    for j = 1:nSuffixes
        out{i, j} = preSufFunc(prefixes{i}, suffixes{j});
    end;
end;
end