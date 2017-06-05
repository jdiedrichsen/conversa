function varargout = cnv_loadLabelFile(filename)
% Loads a files of labels into a dataframe

% The format we get from dload is:
%   struct with fields:
%          pid: [n×1 double] repeated entry
%          cam: [n×1 double] repeated entry
%          min: [n×1 double] 00
%          sec: [n×1 double] 00
%           ms: [n×1 double] 000
%        Frame: [n×1 double] 000
%   behaviour1: [n×1 double] one hot
%   behaviour2: [n×1 double] one hot
%   behaviour3: [n×1 double] one hot

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
% pid   cam behavior    start	stop    <- fields
% 1001  1   smile		1.432	1.788
% 1001  1   smile		2.3     2.7
% 1001  1   talk        0.015	4.7

FRAME_RATE = 30; % Assumed 30 fps

FIELD_MAP = containers.Map( ...
    {'smiling', 'talking', 'laughing'}, ... % Behaviours
    {'behaviour', 'behaviour', 'behaviour'} ...
    );

dlf = dload(filename); % dlf for dloadFile

% Set time in seconds

if (isfield(fld,'ms')) % Decode by using milliseconds
    
elseif (isfield(fld,'Frame')) % Decode by using frame numbers
    
else % Error - neither frames nor milliseconds provided
    error('Label file does not contain milliseconds or frame numbers');
end;

nEntries = length(dlf.pid);

% Set pid and cam as single entry
% cnv_loadLabelFile.pid = dlf.pid(1); % Set pid
% cnv_loadLabelFile.cam = dlf.cam(1); % Set cam number

% Set pid and cam as array of entries
cnv_loadLabelFile.pid = repmat(dlf.pid(1), 1, nEntries)'; % Set pid array (does not vary)
cnv_loadLabelFile.cam = repmat(dlf.cam(1), 1, nEntries)'; % Set cam number array (does not vary)

dlfFields = fieldnames(dlf);

% Identify and set behaviour fields
behaviourFields = {};
for i = 1:length(dlfFields)
    chkField = dlfFields{i};
    if (isKey(FIELD_MAP, chkField) && strcmp(FIELD_MAP(chkField), 'behaviour') == 1) % Field is a behaviour field
        behaviourFields{end+1} = chkField;
    end;
end;

for i = 1:length(behaviourFields)
    behav = behaviourFields{i};
    % TODO: Go through dlf behaviour field and add ranges where behaviour
    % is 1 to cnv_loadLabelFile
end;

end % cnv_loadLabelFile

function secs = framesToSeconds()
    
end