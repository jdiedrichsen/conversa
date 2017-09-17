function labels = cnv_loadLabelFile(fileName, varargin)
% Loads a files of labels into a dataframe struct
% By Shayaan Syed Ali
% Last updated 05-Jun-17

% The format we get from dload is:
%   struct with fields:
%          pid: [n×1 double] repeated entry
%          cam: [n×1 double] repeated entry
%          min: [n×1 double] 00
%          sec: [n×1 double] 00
%        frame: [n×1 double] 000
%   behaviour1: [n×1 double] one hot
%   behaviour2: [n×1 double] one hot
%   behaviour3: [n×1 double] one hot
% Notes that either ms or frames will be fields, not both

% We want
%   struct with fields: pid, cam, start, stop, behaviour (e.g. smiling,
%   laughing, etc.)

% What we want (Jorn):
% D.behavior 
% D.start
% D.stop
% D.pid
% D.cam
% 
% pid   cam behavior    start	end    <- fields
% 1001  1   smile		1.432	1.788
% 1001  1   smile		2.3     2.7
% 1001  1   talk        0.015	4.7

% ADD: Alternatively, we can load in a one hot encoding with timestamps for each
% frame

global FRAME_RATE;
FRAME_RATE = 30; % Assumed 30 fps for frame encoding

behaviors={ 'smile', 'talk', 'laugh','video'}; 

labels = [];

dlf = dload(fileName); % dlf for dloadFile
dlf.video = ones(length(dlf.frame),1); 


pid = dlf.pid(1); % Set pid
cam = dlf.cam(1); % Set cam number array (does not vary)
dlfFields = fieldnames(dlf);

% Identify behaviour fields
behaviourFields = {};
for i = 1:length(dlfFields)
    if (any(strcmp(dlfFields{i},behaviors))) 
        behaviourFields{end+1} = dlfFields{i};
    end
end

entryNo = 1;
for i = 1:length(behaviourFields)
    % Go through dlf behaviour field and add ranges where behaviour is 1
    behavName = behaviourFields{i};
    behav = dlf.(behavName);
    behavLength = length(behav);
    behavStartI = 1;
    while (behavStartI < behavLength)
        while (behavStartI < behavLength && behav(behavStartI) ~= 1) % Go to next 1
           behavStartI = behavStartI+1;
        end
        behavEndI = behavStartI;
        while (behavEndI < behavLength && behav(behavEndI) == 1) % Go to end of '1' state (i.e. end of on state')
           behavEndI = behavEndI+1;
        end
        % Add to label struct
        labels.pid(entryNo, 1) = pid;
        labels.cam(entryNo, 1) = cam;
        labels.behaviour(entryNo, 1) = {behavName}; % As cell array
%         labels.behaviour(entryNo, 1:length(behavName)) = behavName; % As char array
        labels.start(entryNo, 1) = toSeconds(dlf, behavStartI);
        labels.end(entryNo, 1) = toSeconds(dlf, behavEndI);
        entryNo = entryNo+1;
        behavStartI = behavEndI+1;
    end
end

end % cnv_loadLabelFile

function outSecs = toSeconds(dlf, i)
global FRAME_RATE;
outSecs = (60)*dlf.min(i) + dlf.sec(i)+ (1/FRAME_RATE)*dlf.frame(i);
end

