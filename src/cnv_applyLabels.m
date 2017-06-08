function varargout = cnv_applyLabels(trackingData, labels)

excludeFields = {'timestamp', 'istracked', 'bodyid'};

start = firstChangeI(trackingData, 'exclude', excludeFields);

end

% Finds first point index of change in a data struct's fields
function firstChangeI(data, varargin)
args = cnv_getArgs(varargin);
fields = fieldnames(data);
% Remove fields to exclude
if(isfield(args, 'exclude'))
    fields = setdiff(fields, args.exclude);
end;
% Iterate through all fields until a change is found or the end is reached 

end % firstChangeI