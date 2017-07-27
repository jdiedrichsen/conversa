function labelledDf = cnv_applyLabels(trackingData, labels)
% TODO: Documentation

excludeFields = {'timestamp', 'istracked', 'bodyid'};

% Add trackingData fields after they start tracking (after first point of
% change)
start = cnv_firstChangeI(trackingData, 'exclude', excludeFields);
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

% Transform start and end vectors to indices from timestamps
labels.start = timestampIndex(labels.start);
labels.end = timestampIndex(labels.end);

nBehavs = length(labels.behaviour);
for i = 1:nBehavs
    behavName = labels.behaviour{i};
    labelledDf.(behavName)(1:nFrames, 1) = false;
end;
for i = 1:nBehavs
    behavName = labels.behaviour{i};
    startI = labels.start(i);
    endI = labels.end(i);
    labelledDf.(behavName)(startI:endI, 1) = true;
end;
end % cnv_applyLabels

% Gets the index of a given timestamp
function index = timestampIndex(time)
    index = round(30*time + 1);
end