% Converts a struct to a matrix, assumes all included field lengths are equal
% Varargin options:
%	includefields
%		A cell list of fields to include or the string 'all' to include all
%		fields, 'all' is assumed by default
%	excludefields
%		A cell list of fields to exclude or the string 'none' so no fields
%		are excluded, 'none' is assumed by default
function outMatrix = cnv_struct2Matrix(inStruct, varargin)

% Load options
defaults = struct( ...
	'includefields', 'all', ...
	'excludefields', 'none' ...
	);
optionArgs = cnv_getArgs(defaults, varargin)

% Add fields to include
fields = fieldnames(inStruct); % Use all fields in 'all' case
if (~(ischar(optionArgs.includefields) && strcmp(optionArgs.includefields, 'all'))) % Otherwise modified as required
	fields = optionArgs.includefields
end
% Remove fields to exclude
if (~(ischar(optionArgs.excludefields) && strcmp(optionArgs.excludefields, 'none'))) % Otherwise modified as required
	excludeFields = optionArgs.excludefields;
else
	excludeFields = {};
end
fields = setdiff(fields, excludeFields);

% Add fields to matrix with columns as fields and rows as entries
nFields = length(fields);
outMatrix = zeros(length(inStruct.(fields{1})), nFields);
for i = 1:nFields
	outMatrix(:,i) = inStruct.(fields{i});
end

end