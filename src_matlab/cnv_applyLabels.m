function labelledDf = cnv_applyLabels(trackingData, labels)
% function labelledDf = cnv_applyLabels(trackingData, labels)
% 
% Joins a tracking file with the corresponding label file 
% 
excludeFields = {'timestamp', 'istracked', 'bodyid'};

% Add trackingData fields after they start tracking (after first point of
% change)
start = cnv_firstChangeI(trackingData, 'exclude', excludeFields);
fprintf('removed %d frames\n',start);
labelledDf=getrow(trackingData,start:length(trackingData.timestamp)); 

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

% Check length of the tracking file against video length 
i=find(strcmp(labels.behaviour,'video')); 
fprintf('Differences in Frames: %d\n',nFrames-labels.end(i));
labels=getrow(labels,~strcmp(labels.behaviour,'video')); % Remove video field 

behavNames = unique(labels.behaviour); 
nBehavs = length(behavNames);
for i = 1:nBehavs
    labelledDf.(behavNames{i})(1:nFrames, 1) = false;
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