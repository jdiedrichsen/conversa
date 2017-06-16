function varargout = cnv_plotSpectro(plotData, varargin)
% TODO: documentation

optionArgs = cnv_getArgs(varargin);

% Set time range for plotting
% This code was taken from cnv_plotTrackingData, a future change will
% allow for easier loading of optional parameters
time = plotData.timestamp;
startFrame = cnv_firstChangeI(plotData, 'exclude', {'timestamp', 'istracked', 'bodyid'});
endFrame = length(time);
if (isfield(optionArgs, {'start'}))
    startFrame = optionArgs.start;
end;
if (isfield(optionArgs, {'end'}))
    endFrame = optionArgs.end;
end;
startTime = indexToTime(startFrame);
endTime = indexToTime(endFrame);
if (isfield(optionArgs, {'starttime'}))
    startTime = optionArgs.starttime;
end;
if (isfield(optionArgs, {'endtime'}))
    endTime = optionArgs.endtime;
end;
range = startFrame:endFrame;
time = time(range); % Restrict to range



end