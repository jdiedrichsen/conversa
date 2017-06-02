function varargout = cnv_loadLabelFile(filename)

% d.smiling = {[100 1200], [1560 1567]}
% 
% 
% D.behavior 
% D.start
% D.stop
% D.pid
% D.sid 
% 
% behavior	start	stop	pid
% smile		1.432	1.783	1
% smile		2.3	2.7	1
% talk		0.015	4.7	1

 


% The format we get from dload is:
%   struct with fields:
%          pid: [n×1 double] repeated entry
%          cam: [n×1 double] repeated entry
%          min: [n×1 double] 00
%          sec: [n×1 double] 00
%           ms: [n×1 double] 000
%   behaviour1: [n×1 double] one hot
%   behaviour2: [n×1 double] one hot
%   behaviour3: [n×1 double] one hot

% We want
%   struct with fields: pid, cam, start, stop, behaviour (e.g. smiling,
%   laughing, etc.)

dlf = dload(filename); % dlf for dloadFile

