function varargout = cnv_plotLinear(plotData, varargin)
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
%   start           Frame to start at
%   end             Frame to end at
%   starttime       Time to start at
%   endtime         Time to end at
%   nrows           Number of rows in plot
%   ncols           Number of columns in plot
%   plotgroups      Groups to plot
%                   Usage example: 'plotgroups', {'Neck', 'Head'}
%   plotfields      Fields to plots together
%                   Usage example: plotfields, {'Neck X Position and Rotation', {'neckPosX', 'neckRotX'}, 'Smile', {'Smile_L', 'Smile_R'} }
%   timeunits       (WIP) Can either set timeunits to frames or seconds
%                   Usage example: 'timeunits', 'seconds'
%   labels          Label dataframe to plot
%   annotations     (WIP) Whether to add annotations to labels or not
%   plottype        Allows for linear or spectrogram plotting
% By Shayaan Syed Ali
% Last updated 30 May 17

% TODO: Refactor varargin parameter loading, e.g. by iterating through
% varargin and setting params as needed

% We can use inputParser for type checking, but this is slower
% Jorn mentioned sometime like 'varargin options', I was unable to find a
% reference to built in Matlab functions which did anything like this

% Default grouping of data ADD: change into default struct containing all
% default option args
    defaultPlotMap = containers.Map({'all', 'neck', 'head', 'brow', 'eyes', 'cheek', 'lips', 'smile', 'frown', 'jaw'}, { ...
    {'neckposx', 'neckposy', 'neckposz', 'neckrotx', 'neckroty', 'neckrotz', 'headposx', 'headposy', 'headposz', 'headrotx', 'headroty', 'headrotz', 'brow_up_l', 'brow_up_r', 'brow_down_l', 'brow_down_r', 'eye_closed_l', 'eye_closed_r', 'cheek_puffed_l', 'cheek_puffed_r', 'lips_pucker', 'lips_stretch_l', 'lips_stretch_r', 'lip_lower_down_l', 'lip_lower_down_r', 'smile_l', 'smile_r', 'frown_l', 'frown_r', 'jaw_l', 'jaw_r', 'jaw_open'}, ...
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

% Initialize paramters from input option arguments as needed - this will be
% refactored
optionArgs = [];
if (nargin>0)
    optionArgs = cnv_getArgs(varargin); % Load args
end;

if (~exist('plotData', 'var'))
    plotData = cnv_loadTrackingData('C:\Users\Shayn\Documents\Work\AI Research\conversa\data\par1001Cam2\cam2par1001.txt');
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
    plotFieldArgs = cnv_getArgs(optionArgs.plotfields);
    plotFieldNames = fieldnames(plotFieldArgs);
    nPlotFields = length(plotFieldNames);
    for i = 1:nPlotFields
        fieldName = plotFieldNames{i};
        plotMap(fieldName) = plotFieldArgs.(plotFieldArgs);
    end;
end;

% TODO: Refactor time ranging
% Get timestamps
time = plotData.timestamp;
% Set plotting range
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

% TODO: add styling options for plotting

% TODO: add default colours for labels
% TEMP: colour map coded for behaviours
labelColorMap = containers.Map({'smiling', 'laughing', 'talking'}, {...
    'r', ...
    'g', ...
    'b' ...
    });

annotations = true;
if(isfield(optionArgs, {'annotations'}))
    annotations = optionArgs.annotations;
end;

% Load label arg
labels = [];
labelsIncl = false;
if (isfield(optionArgs, {'labels'}))
    labels = optionArgs.labels;
    labelsIncl = true;
end;

% Load plottype
plotType = 'linear';
if(isfield(optionArgs, {'plottype'}))
    plotType = optionArgs.plottype;
end;

% Plot data

fig = figure; % Create figure

% Go through groups and plot
plotGroups = keys(plotMap);

% Get maxNFields for subplotting with plottype spectro
maxNFields = 0;
for i = 1:length(plotGroups)
    maxNFields = max(maxNFields, length(plotMap(plotGroups{i})));
%     disp(horzcat(plotGroups{i}, ' ', num2str(length(plotMap(plotGroups{i})))))
end;

% Set number of horizontal and vertical cells for subplots
% Default format is a column
nFigRows = length(plotMap);
if (isfield(optionArgs, {'nrows'}))
    nFigRows = optionArgs.nrows;
end;
nFigCols = 1;
if (isfield(optionArgs, {'ncols'}))
    nFigCols = optionArgs.ncols;
elseif (strcmp(plotType, 'spectro') == 0)
    nFigCols = maxNFields;
end;

for i = 1:min(nFigCols*nFigRows, length(plotGroups)) % Iterate through groups, stop when no more plot positions or all groups plotted
    % Plot tracking data
    plotGroup = plotGroups{i};
    fields = plotMap(plotGroup);
    % Plot fields in groups
    % TODO: Refactor to function call
    nFields = length(fields);
    switch plotType
        case 'linear'
            subplot(nFigRows, nFigCols, i)
            for j = 1:nFields
                plot(time, plotData.(fields{j})(range)); hold on;
            end;
        case 'spectro'
            for j = 1:nFields
                subplot(nFigRows, maxNFields, j + (i-1)*maxNFields)
                spectrogram(plotData.(fields{j})(range), 'yaxis');
                t = title(fields{j}); hold on;
                set(t,'interpreter','none')
            end;
        otherwise
            error('Invalid plottype');
    end;
    % Get axis data and set axis limits
    yLimits = ylim;
    lowerYLim = yLimits(1);
    upperYLim = yLimits(2);
    axis([startTime endTime lowerYLim upperYLim]);
    % Plot labels
    if (labelsIncl)
        behavs = labels.behaviour;
        nLabels = length(behavs);
        for j = 1:nLabels
            startT = startTime + (labels.start(j));
            endT = startTime + (labels.end(j));
            if (endT-startT > 0) % Check that the plot is valid and TODO: within plotting range
                % TODO: Adjust startT and endT appropriately when they are out
                % of plot range
                patch('Faces', 1:4, 'Vertices', [startT lowerYLim; endT lowerYLim; endT upperYLim; startT upperYLim], ...
                    'FaceColor', labelColorMap(behavs{j}), 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
                % WIP: Add label annotation
    %             if (annotations)
    %                 annotation('textbox', [startT lowerYLim endT-startT 1], ...
    %                 'String', labels.behaviour{j}); % Position must be percentage, e.g. [0.3 0.4 0.1 0.2]
    %             end;
            end;
        end;
    end;
    % Plot fields in groups on top of labels
    % TODO: Refactor to function call, currently must copy code from above
    % -_-
    if (labelsIncl)
        nFields = length(fields);
        switch plotType
            case 'linear'
                for j = 1:nFields
                    plot(time, plotData.(fields{j})(range)); hold on;
                end;
                    % Add legend and axis labels
                    l=legend(fields); % ADD: hide/how legend option
                    set(l,'interpreter','none'); % Prevents interpretation of underscores as subscripts in legends
                %     xlabel('time (seconds)'); % ADD: different time unit option and hide xlabel option
                %     ylabel('magnitude (a.u.)'); % ADD: hide ylabel option
                    title(plotGroup);
            case 'spectro'
%                 end;
            otherwise
                error('Invalid plottype');
        end;
    end;
end;

end % cnv_plotTracking

% Converts from a frame index to a time
function time = indexToTime(index)
    time = (index-1)/30;
end

% Gets the index of a given timestamp
function index = timeToIndex(time)
    index = round(30*time + 1);
end