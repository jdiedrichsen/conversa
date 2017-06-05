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

global FRAME_RATE;
FRAME_RATE = 30; % Assumed 30 fps for frame encoding

FIELD_MAP = containers.Map( ...
    {'smiling', 'talking', 'laughing'}, ... % Behaviours
    {'behaviour', 'behaviour', 'behaviour'} ...
    );

dlf = dload(filename); % dlf for dloadFile

% Figure out which conversion function to use (frames or seconds)
toSeconds = [];
if (isfield(dlf,'ms')) % Decode by using milliseconds
    toSeconds = str2func(millisToSeconds);
elseif (isfield(dlf,'frame')) % Decode by using frame numbers
    toSeconds = str2func(framesToSeconds);
else % Error - neither frames nor milliseconds provided
    error('Label file does not contain milliseconds or frame numbers');
end;

pid = dlf.pid(1); % Set pid
cam = dlf.cam(1); % Set cam number array (does not vary)

dlfFields = fieldnames(dlf);

% Identify behaviour fields
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
    behavStartI = 1;
    while (behavStartI < behavLength)
        while (behavStartI < behavLength && behav(behavStartI) ~= 1) % Go to next 1
           behavStartI = behavStartI+1;
        end;
        behavEndI = behavStartI+1;
        while (behavEndI < behavLength && behav(behavStartI) == 1) % Go to end of '1' state (i.e. end of on state')
           behavEndI = behavEndI+1;
        end;
        % Add to label struct
        cnv_loadLabelFile.pid(entryNo) = pid;
        cnv_loadLabelFile.cam(entryNo) = cam;
        cnv_loadLabelFile.behaviour(entryNo) = behavName;
        cnv_loadLabelFile.start(entryNo) = toSeconds(dlf, behavStartI); % WIP TODO: change to time
        cnv_loadLabelFile.end(entryNo) = toSeconds(dlf, behavEndI); % WIP TODO: change to time
        entryNo = entryNo+1;
    end;
end;

end % cnv_loadLabelFile

function outSecs = framesToSeconds(dlf, i)
global FRAME_RATE;
outSecs = (60)*dlf.min(i) + dlf.sec(i)+ (1/FRAME_RATE)*dlf.frame(i);
end

function outSecs = millisToSeconds(dlf, i)
outSecs = (60)*dlf.min(i) + dlf.sec(i)+ (1/1000)*dlf.frame(i);
end