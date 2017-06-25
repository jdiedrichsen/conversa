% Loads a struct with the parameters
% By Shayaan Syed Ali
% Last updated 19-Jun-17
function outArgs = cnv_getArgs(defaults, vargs) % TODO: Change signature to cnv_getArgs(defaults, vargin) and set args = defaults, update as needed
if (mod(length(vargs), 2) ~= 0)
    error('Must have an equal number of fields and values');
end;
outArgs = defaults;
for i = 1:2:length(vargs)
    outArgs.(vargs{i}) = vargs{i+1}; % TODO: Check if field exists in defaults before adding, must refactor code which calls this to pass defaults through defaults first
end;
end % cnv_fieldValuePairs