function D = cnv_loadFaceData(fileName)
% Loads a tab delimited text file of motion tracking data into a struct
% Usage
%   D=cnv_loadTrackingData('fileName')
% Description
%   Reads and stores data from a Kinect face-tracking data text file
%   The data is stored in a dataframe, a structure with a 2D array of fields. 
%   By Shayaan Syed Ali
%   Last updated 09-Sep-17 joern.diedrichsen 

TOP_HEADER_LN = '==========';
N_HEADER_LNS = 3;


% Open File 
fid = fopen(fileName,'r'); % Open the file for reading
if (fid == -1) % Indicates file not found
    error('Error: Did not find file: %s',fileName);
end

% Read in the metadata file from file header
% Ignore some metadata lines in file
for i = 1:7 
    fgetl(fid);
end

% Get the number of frames from metadata
nFrameLnStr = fgetl(fid); % The line in the file containing the number of frames
nFrames = str2double(nFrameLnStr(regexp(nFrameLnStr, '(\d*)$'):end)); % Finds number in line by searching from back of string

% Go to first face
% Read header of first first 
for i=1:4
    fgetl(fid); 
end; 

% Set up header 
Header=fgetl(fid);
H={};
Head=Header;
while length(Head)>0
    [r,Head]=strtok(Head);
    if (~isempty(r))
        H{end+1}=r;
    end; 
end;
nCols = length(H); 
fclose(fid);

% Read data 
A=textread(fileName,'%f','headerlines',13);
if (length(A)~=(nCols*nFrames)) 
    error('file has not the right size'); 
end; 
A = reshape(A,nCols,nFrames)'; 
for i=1:nCols 
    D.(H{i}) = A(:,i); 
end; 

