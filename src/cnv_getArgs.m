% Loads a struct with the parameters
% By Shayaan Syed Ali
% Last updated 05-Jun-17
function args = cnv_getArgs(vargs)
if (mod(length(vargs), 2) ~= 0)
    error('Must have an equal number of fields and values');;
end;
args=[];
for i = 1:2:length(vargs)
    args.(vargs{i}) = vargs{i+1};
end;
end % cnv_fieldValuePairs