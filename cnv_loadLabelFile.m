function varargout = cnv_loadLabelFile(filename)
% Loads a files of labels into a dataframe
% By Shayaan Syed Ali
% Last updated 05-Jun-17

% The format we get from dload is:
%   struct with fields:
%          pid: [n×1 double] repeated entry
%          cam: [n×1 double] repeated entry
%          min: [n×1 double] 00
%          sec: [n×1 double] 00
%           ms: [n×1 double] 000
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

FRAME_RATE = 30; % Assumed 30 fps for frame encoding

FIELD_MAP = containers.Map( ...
    {'smiling', 'talking', 'laughing'}, ... % Behaviours
    {'behaviour', 'behaviour', 'behaviour'} ...
    );

dlf = dload(filename); % dlf for dloadFile

% % Convert time to seconds
% time = []; % Time vector in floating pt seconds
% if (isfield(dlf,'ms')) % Decode by using milliseconds
%     time = millisToSeconds(dlf.min, dlf.sec, dlf.ms);
% elseif (isfield(dlf,'frame')) % Decode by using frame numbers
%     time = framesToSeconds(dlf.min, dlf.sec, dlf.frame, FRAME_RATE);
% else % Error - neither frames nor milliseconds provided
%     error('Label file does not contain milliseconds or frame numbers');
% end;

pid = dlf.pid(1); % Set pid
cam = dlf.cam(1); % Set cam number array (does not vary)

dlfFields = fieldnames(dlf);

% Identify and set behaviour fields
behaviourFields = {};
for i = 1:length(dlfFields)
    chkField = dlfFields{i};
    if (isKey(FIELD_MAP, chkField) && strcmp(FIELD_MAP(chkField), 'behaviour') == 1) % Field is a behaviour field
        behaviourFields{end+1} = chkField;
    end;
end;

entryNo = 1;
for i = 1:length(behaviourFields)
    % Go through dlf behaviour field and add ranges where behaviour is 1
    behavName = behaviourFields{i};
    behav = dlf.(behavName);
    behavLength = length(behav);
    behavStart = 1;
    while (behavStart < behavLength)
        while (behavStart < behavLength && behav(behavStart) ~= 1) % Go to next 1
           behavStart = behavStart+1;
        end;
        behavEnd = behavStart+1;
        while (behavEnd < behavLength && behav(behavStart) == 1) % Go to end of '1' state (i.e. end of on state')
           behavEnd = behavEnd+1;
        end;
        % Add to label struct
        cnv_loadLabelFile.pid(entryNo) = pid;
        cnv_loadLabelFile.cam(entryNo) = cam;
        cnv_loadLabelFile.behaviour(entryNo) = behavName;
        cnv_loadLabelFile.start(entryNo) = behavStart; % WIP TODO: change to time
        cnv_loadLabelFile.end(entryNo) = behavEnd; % WIP TODO: change to time
        entryNo = entryNo+1;
    end;
end;

end % cnv_loadLabelFile

function outSecs = framesToSeconds(min, sec, frameNo, frameRate) % Assumes 30 fps
    outSecs = (60)*min + sec + (1/frameRate)*frameNo;
end

function outSecs = millisToSeconds(min, sec, ms)
    outSecs = (60)*min + sec + (1/1000)*ms;
end