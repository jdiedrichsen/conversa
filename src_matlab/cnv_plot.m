function out = cnv_plot(T, varargin)
% function out = cnv_plot(T, varargin)
% plots tracking data in subplots with various parameters
% Input
%   Dataframe from cnv_applyLabels
% Varargin
%   starttime       Time to start at
%   endtime         Time to end at
%   plotgroups      Groups to plot
%                   Usage example: 'plotgroups', {'neck', 'head'}
%   plotmap         New map that defines which fields are presented
%                   together
%   timeunits       (WIP) Can either set timeunits to frames or seconds
%                   Usage example: 'timeunits', 'seconds'
%   labels          Label dataframe to plot
%   annotations     (WIP) Whether to add annotations to labels or not
%   plottype        Allows for linear or spectrogram plotting
% By Shayaan Syed Ali
% Last updated 19-Jun-17

% TODO: Refactor varargin parameter loading, e.g. by iterating through
% varargin and setting params as needed

% We can use inputParser for type checking, but this is slower
% Jorn mentioned sometime like 'varargin options', I was unable to find a
% reference to built in Matlab functions which did anything like this

% Default grouping of data ADD: change into default struct containing all
% default option args
plotmap = containers.Map( ...
    {'neck', 'head', 'brow', 'eyes', 'cheek', 'lips', 'smile', 'frown', 'jaw'}, { ...
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

labelColorMap = containers.Map({'smile', 'laugh', 'talk'}, {'r','k','b'});

% Set Default
starttime = min(T.timestamp);
endtime  = max(T.timestamp);
plotgroups = keys(plotmap);
timeunits = 'seconds';
labels   = {};
plottype = 'timeseries';

% Initialize paramters from input option arguments as needed
vararginoptions(varargin,{'plotgroups','starttime','endtime',...
    'plotfields','timeunits','labels','annotations','plottype'});

% Get timestamps
time = T.timestamp;

% Set plotting range
[~,startFrame] = min(abs(time-starttime)); % Find closest start frame
[~,endFrame] = min(abs(time-endtime)); % Find closest start frame
range = [startFrame:endFrame];

% Plot data
fig = gcf; % Create figure

% Set number of horizontal and vertical cells for subplots
% Default format is a column
nFigRows = length(plotgroups);
nFigCols = 1;

for i = 1:length(plotgroups);
    % Plot tracking data
    fields = plotmap(plotgroups{i});
    nFields = length(fields);
    subplot(nFigRows, nFigCols, i)
    data = [];
    switch plottype
        case 'timeseries'
            for j = 1:nFields
                data = [data T.(fields{j})(range)];
            end;
            plot(time(range),data(range,:));
            set(gca,'XLim',[starttime endtime]);
            l=legend(fields); % ADD: hide/how legend option
            YL = ylim;
            l=legend(fields); % ADD: hide/how legend option
            set(l,'interpreter','none'); % Prevents interpretation of underscores as subscripts in legends
            
            % Plot labels
            for j=1:length(labels)
                cnv_drawPatches(time,T.(labels{j}),labelColorMap(labels{j}));
            end;
            title(plotgroups{i});
            
            
        case 'spectro'
            for j = 1:nFields
                subplot(nFigRows, nFigCols, j + (i-1)*maxNFields)
                spectrogram(T.(fields{j})(range), 'yaxis');
                t = title(fields{j}); hold on;
                set(t,'interpreter','none')
            end
        otherwise
            error('Invalid plottype');
    end
end
