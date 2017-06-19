% Loads a struct with the parameters
% By Shayaan Syed Ali
% Last updated 05-Jun-17
function args = cnv_getArgs(inArgs, vargs) % TODO: Change signature to cnv_getArgs(defaults, vargin) and set args = defaults, update as needed
if (mod(length(vargs), 2) ~= 0)
    error('Must have an equal number of fields and values');;
end;
args = inArgs;
for i = 1:2:length(vargs)
    args.(vargs{i}) = vargs{i+1};
end;
end % cnv_fieldValuePairs