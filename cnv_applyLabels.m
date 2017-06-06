function varargout = cnv_applyLabels(trackingData, labels)



end

% Finds first point index of change in a data struct's fields
function firstChangeI(data, varargin)
args = cnv_getArgs(varargin);
fields = fieldnames(data);
% Remove fields to exclude
if(isfield(args, 'exclude'))
    rmCells(fields, args.exclude);
end;
% TODO: Iterate through all fields until a change is found or the end is reached 

end

% Removes all elements in rm from arr
function rmCells(arr, rm)
    
end