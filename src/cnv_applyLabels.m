function labelledDf = cnv_applyLabels(trackingData, labels)
% TODO: Documentation

excludeFields = {'timestamp', 'istracked', 'bodyid'};

% Add trackingData fields after they start tracking (after first point of
% change)
start = firstChangeI(trackingData, 'exclude', excludeFields);
trackingFields = fieldnames(trackingData);
nTrackingFields = length(trackingFields);
for i = 1:nTrackingFields
    fieldName = trackingFields{i};
    labelledDf.(fieldName) = trackingData.(fieldName)(start:end);
end;
% Zero timestamps to true start
labelledDf.timestamp = labelledDf.timestamp - labelledDf.timestamp(1);
% There seems to be additional time adjustments required here as the
% timestamps currently have a length difference of almost 1 second
nFrames = length(labelledDf.timestamp);

% Add labels
% Set all values to false then update from labels
nBehavs = length(labels.behaviour);
for i = 1:nBehavs
    behavName = labels.behaviour{i};
    labelledDf.(behavName)(1:nFrames, 1) = false;
end;
for i = 1:nBehavs
    behavName = labels.behaviour{i};
    startI = timestampIndex(labels.start(i));
    endI = timestampIndex(labels.end(i));
    labelledDf.(behavName)(startI:endI, 1) = true;
end;
end % cnv_applyLabels

% Finds first point index of change in a data struct's fields
function changeI = firstChangeI(data, varargin)
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

function index = timestampIndex(time)
    index = round(30*time + 1);
end