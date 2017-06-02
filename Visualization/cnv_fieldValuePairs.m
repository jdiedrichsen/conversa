% Loads a struct with the parameters
function args = cnv_fieldValuePairs(vargs)
if (mod(length(vargs), 2) ~= 0)
    error('Must have an equal number of fields and values');;
end;
args=[];
for i = 1:2:length(vargs)
    args.(vargs{i}) = vargs{i+1};
end;