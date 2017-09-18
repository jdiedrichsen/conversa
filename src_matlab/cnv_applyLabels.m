function labelledDf = cnv_applyLabels(trackingData, labels)
% function labelledDf = cnv_applyLabels(trackingData, labels)
% 
% Joins a tracking file with the corresponding label file 
% 
excludeFields = {'timestamp', 'istracked', 'bodyid'};
alignAt = 'start'; 
verbose = 1; 

% Add trackingData fields after they start tracking (after first point of
% change)
start = cnv_firstChangeI(trackingData, 'exclude', excludeFields);
if (verbose) 
    fprintf('removed %d frames\n',start);
end; 
labelledDf=getrow(trackingData,start:length(trackingData.timestamp)); 

% Check length of the tracking file against video length 
nFrames = length(labelledDf.timestamp);
i=find(strcmp(labels.behaviour,'video')); 
videolength = labels.end(i); 
tracklength = labelledDf.timestamp(end)-labelledDf.timestamp(1); 
if (verbose) 
    fprintf('Differences in sec: %2.3f\n',tracklength-videolength);
end; 
labels=getrow(labels,~strcmp(labels.behaviour,'video')); % Remove video field 

% Given slightly different length, you can align in the front or back 
% Front 
switch (alignAt)
    case 'start' 
        labelledDf.timestamp = labelledDf.timestamp - labelledDf.timestamp(1);
    case 'end' 
        labelledDf.timestamp = labelledDf.timestamp - labelledDf.timestamp(end)+videolength;
end; 

% Look up the indeicdes for the timestamps
labels.startI = timestampIndex(labelledDf.timestamp,labels.start);
labels.endI   = timestampIndex(labelledDf.timestamp,labels.end);

behavNames = unique(labels.behaviour); 
nBehavs = length(behavNames);
for i = 1:nBehavs
    labelledDf.(behavNames{i})(1:nFrames, 1) = false;
end;
for i = 1:length(labels.behaviour); 
    behavName = labels.behaviour{i};
    labelledDf.(behavName)(labels.startI(i):labels.endI(i), 1) = true;
end;
end % cnv_applyLabels

% Gets the index of a given timestamp
function index = timestampIndex(time,timestamp)
    for i=1:length(timestamp) 
        [~,index(i,1)] = min(abs(time-timestamp(i))); % Find closest timestamp
    end; 
end