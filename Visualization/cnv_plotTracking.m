function varargout = cnv_plotTracking(plotData, varargin)
% CNV_PLOTTRACKING: plots tracking data in subplots with various parameters
% Input
%   TODO
% Output
%   TODO
% Usage
%   TODO
% Description
%   TODO
% Argument list
%   labels          Struct of labels to use
%                   Usage example: 'labels', dload(
%   start           Frame to start at
%   end             Frame to end at
%   starttime       Time to start at (WIP)
%   endtime         Time to end at (WIP)
%   nrows           Number of rows in plot
%   ncols           Number of columns in plot
%   plotgroups      Groups to plot
%                   Usage example: 'plotgroups', {'Neck', 'Head'}
%   plotfields      Fields to plots together
%                   Usage example: plotfields, {'Neck X Position and Rotation', {'neckPosX', 'neckRotX'}, 'Smile', {'Smile_L', 'Smile_R'} }
%   timeunits       Can either set timeunits to frames or seconds (WIP)
%                   Usage example: 'timeunits', 'seconds'
% Last updated 30 May 17

% We can use inputParser for type checking, but this is slower
% Jorn mentioned sometime like 'varargin options', I was unable to find a
% reference to built in Matlab functions which did anything like this

% Default grouping of data
    defaultPlotMap = containers.Map({'neck', 'head', 'brow', 'eyes', 'cheek', 'lips', 'smile', 'frown', 'jaw'}, { ...
    {'neckposx', 'neckposy', 'neckposz', 'neckrotx', 'neckroty', 'neckrotz'}, ...
    {'headposx', 'headposy', 'headposz', 'headrotx', 'headroty', 'headrotz'}, ...
    {'brow_up_l', 'brow_up_r', 'brow_down_l', 'brow_down_r'}, ...
    {'eye_closed_l', 'eye_closed_r'}, ...
    {'cheek_puffed_l', 'cheek_puffed_r'}, ...
    {'lips_pucker', 'lips_stretch_l', 'lips_stretch_r', 'lip_lower_down_l', 'lip_lower_down_r'}, ...
    {'smile_l', 'smile_r'}, ...
    {'frown_l', 'frown_r'}, ...
    {'jaw_l', 'jaw_r', 'jaw_open'} ...
 });

% Initialize paramters from input option arguments as needed

optionArgs=[];

if (nargin>0)
    optionArgs = cnv_loadArgs(varargin); % Load args
end;

% Remove and add plot groups as needed
if (isfield(optionArgs, {'plotgroups'}))
    plotGroups = optionArgs.plotgroups;
    plotMap = containers.Map();
    for i = 1:length(plotGroups)
        plotMap(plotGroups{i}) = defaultPlotMap(plotGroups{i});
    end;
else % Initialize defaults
    plotMap = defaultPlotMap;
end;
if (isfield(optionArgs, {'plotfields'}))
    plotFieldArgs = cnv_loadArgs(optionArgs.plotfields);
end;

% Get timestamps
time = plotData.timestamp;

% Set plotting range
startFrame = 1;
endFrame = length(time);
if (isfield(optionArgs, {'start'}))
    startFrame = optionArgs.start;
end;
if (isfield(optionArgs, {'end'}))
    endFrame = optionArgs.end;
end;
range = startFrame:endFrame;

% Set number of horizontal and vertical cells for subplots
% Default format is a column
nFigRows = size(plotMap,1);
if (isfield(optionArgs, {'nrows'}))
    nFigRows = optionArgs.nrows;
end;
nFigCols = 1;
if (isfield(optionArgs, {'ncols'}))
    nFigCols = optionArgs.ncols;
end;

% TODO: add styling options for plotting

% Load labels


% Plot data

fig = figure; % Create figure

% Go through groups and plot
plotGroups = keys(plotMap);
for i = 1:min(nFigCols*nFigRows, size(plotGroups,2)) % Iterate through groups, stop when no more plot positions or all groups plotted
    plotGroup = plotGroups{i};
    subplot(nFigRows, nFigCols, i)
    fields = plotMap(plotGroup);
    % Plot fields in groups
    for j = 1:length(fields)
        plot(time(range), plotData.(fields{j})(range)); hold on;
    end;
    l=legend(fields);
    set(l,'interpreter','none'); % Prevents interpretation of underscores as subscripts in legends
    % Plot labels (TEMP WIP)
%    for i = 1:2:length(labels)
%        line(labels{i+1}, get(gca, 'ylim'));
%    end;
    hold off;
    xlabel('time (frames)');
    ylabel('magnitude (a.u.)');
    title(plotGroup);
end;

% If labels given, use drawline to draw vertical lines at onset and offset of each expression 

% Full customizability

% Loads a struct with the parameters
function args = cnv_loadArgs(vargs)
args=[];
for i = 1:2:length(vargs)
    args.(vargs{i}) = vargs{i+1};
end;