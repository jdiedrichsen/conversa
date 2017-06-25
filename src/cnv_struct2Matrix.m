function out = cnv_struct2Matrix(inStruct, varargin)
% Converts a struct to a matrix
% Varargin options:
%	includefields
%		A cell list of fields to include or the string 'all' to include all
%		fields, 'all' is assumed by default
%	excludefields
%		A cell list of fields to exclude. Assumes an empty list by default

defaultOptions = struct( ...
	'includefields', 'all', ...
	'excludefields', {} ...
	);

optionArgs = cnv_getArgs(defaultOptions, varargin);

% Add 
fields = fieldname(inStruct);
if(ischar(optionArgs.includefields) && strcmp(optionArgs.includefields, 'all')) % Use all fields in 'all' case
	
else
	
end;

end