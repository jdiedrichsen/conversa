% Finds first point index of change in a data struct's fields
function changeI = cnv_firstChangeI(data, varargin)
% TODO: Documentation
args = cnv_getArgs(varargin);
fields = fieldnames(data);
% Remove fields to exclude
if(isfield(args, 'exclude'))
    fields = setdiff(fields, args.exclude);
end;
% Ensure that there are fields to search
if (isempty(fields))
    error('No fields to search');
end;
% Iterate through all fields until a change is found or the end is reached
nFields = length(fields);
nEntries = length(data.(fields{1}));
prevVals = zeros(1, nFields);
for i = 1:nFields
    prevVals(i) = data.(fields{i})(1);
end;
nextVals = zeros(1, nFields);
for i = 2:nEntries
    for j = 1:nFields
        nextVals(j) = data.(fields{j})(i);
    end;
    simVec = prevVals == nextVals; % Similarity vector
    for k = 1:nFields
        if (simVec(k) == 0)
            changeI = i;
            return;
        end;
    end;
    prevVals = nextVals;
    nextVals = zeros(1, nFields);
end;
changeI = -1;
return ;  % No change found
end % firstChangeI