function cnvTrackingData = cnv_loadTrackingFile(varargin)
% CNV_DLOAD: loads a tab delimited text file of motion tracking data
%       into memory
% Usage
%   CnvTrackingData=cnvLoad('fileName')
%   CnvTrackingData=cnvLoad()
% Description
%   Reads and stores data from a Kinect face-tracking data text file
%   The data is stored in a structure with a 2D array of fields, with the
%   ith field array belonging to the ith face
%   If no fileName argument is provided a default file is read from
% By Shayaan Syed Ali
% Last updated 26 May 2017

DEFAULT_FILE = 'C:\Users\Shayn\Documents\Work\AI Research\conversa\Test Data\par1001Cam1\cam1par1001.txt';
TOP_HEADER_LN = '==========';
N_HEADER_LNS = 3;
% We use the above since this entry delimits each face (where i is replaced by the faceNo):
% ==========
%  Face: i
% ==========
cnvTrackingData = [];
% Load file location
if (nargin == 1)
    fileName = varargin{1};
else
    fileName = DEFAULT_FILE;
end;
fid = fopen(fileName,'r'); % Open the file for reading
if (fid == -1) % Indicates file not found
    fprintf('Error: Did not find file\n');
    return;
end;
% Read in the metadata file from file header
% Ignore some metadata lines in file
for i = 0:6
    fgetl(fid);
end;
% Get the number of frames from metadata
nFrameLnStr = fgetl(fid); % The line in the file containing the number of frames
nFrames = str2double(nFrameLnStr(regexp(nFrameLnStr, '(\d*)$'):end)); % Finds number in line by searching from back of string
% Go to first face
nextL = fgetl(fid);
while(~feof(fid) && (isempty(nextL) || strcmp(nextL, TOP_HEADER_LN) ~= 1))
    nextL = fgetl(fid);
end;
faceNo = 0;
% Read until no tracked faces are left (eof)
while(~feof(fid))
    % Ignore face header
    for i = 2:N_HEADER_LNS
        fgetl(fid);
    end;
    faceNo = faceNo+1; % Update the number of faces for indexing
    % Parsing while reading, can be spedup
    % Read the data header
    fieldStr = fgetl(fid); % Read the data header line as a string
    fieldCells = [];
    tempHeaderStr = fieldStr;
    while (~isempty(tempHeaderStr))
        [readHeader, tempHeaderStr] = strtok(tempHeaderStr);
        if (~isempty(readHeader))
            fieldCells{end + 1} = lower(readHeader);
        end;
    end;
    % Allocate for frames
    for j = 1:size(fieldCells,2)
        cnvTrackingData(faceNo).(fieldCells{j}) = zeros(nFrames, 1);
    end;
    
    % Read the frame data
    for i = 1:nFrames
        frameVec = str2num(fgetl(fid));
        for j = 1:size(fieldCells,2)
            cnvTrackingData(faceNo).(fieldCells{j})(i, 1) = frameVec(j);
        end;
    end;    
    % Go to next face
    nextL = fgetl(fid);
    while(~feof(fid) && (~isempty(nextL) && strcmp(nextL, TOP_HEADER_LN) ~= 1))
        nextL = fgetl(fid);
    end;
end;
% End of file is reached
fclose(fid);

end % cnv_loadTrackingFile

% Example file (line numbers on left added)
    %    1  ========================================
    %    2  Exported from Brekel Pro Face 2 v2.23
    %    3  ========================================
    %    4  File format version:		1.0
    %    5  Machinename:				ssc-6430b
    %    6  Licensed to:                Erin Heerey
    %    7  Export date/time:			29/03/2017 15:16:10
    %    8  Number of frames:			8948
    %    9  
    %   10  ==========
    %   11  Face: 1
    %   12  ==========
    %   13  timestamp	isTracked	bodyId	neckPosX	neckPosY	neckPosZ	neckRotX	neckRotY	neckRotZ	headPosX	headPosY	headPosZ	headRotX	headRotY	headRotZ	Brow_Up_L	Brow_Up_R	Brow_Down_L	Brow_Down_R	Eye_Closed_L	Eye_Closed_R	Cheek_Puffed_L	Cheek_Puffed_R	Lips_Pucker	Lips_Stretch_L	Lips_Stretch_R	Lip_Lower_Down_L	Lip_Lower_Down_R	Smile_L	Smile_R	Frown_L	Frown_R	Jaw_L	Jaw_R	Jaw_Open
    %   14  0.00000	1	3	-1.50773	-1.68765	-93.65511	-4.55495	-12.80039	-2.18195	0.68025	7.95688	2.28499	7.39181	14.08011	1.26938	34.80733	31.41938	-0.00008	0.00000	22.06141	15.22832	32.97740	45.05043	0.00000	34.48804	29.06981	8.00485	5.00864	86.16735	80.31444	24.84124	18.23990	0.36605	2.11190	26.19046	
    %   15  0.00000	1	3	-1.50773	-1.68765	-93.65511	-4.55495	-12.80039	-2.18195	0.68025	7.95688	2.28499	7.39181	14.08011	1.26938	34.80733	31.41938	-0.00008	0.00000	22.06141	15.22832	32.97740	45.05043	0.00000	34.48804	29.06981	8.00485	5.00864	86.16735	80.31444	24.84124	18.23990	0.36605	2.11190	26.19046	
    % [...] 
    % 8959  298.20001	1	3	-1.78376	-3.52315	-90.25688	-0.84674	-14.84295	-2.77404	1.11260	8.68608	0.09097	-10.88090	19.09622	8.45653	30.10839	39.39293	0.00642	0.00700	6.31621	3.93789	32.76627	38.67698	39.02547	1.41929	1.43304	0.84354	1.09058	1.92263	1.95327	31.53138	23.63512	0.54235	3.85832	16.00941	
    % 8960  298.23334	1	3	-1.78376	-3.52666	-90.25688	-0.84677	-14.83541	-2.76711	1.10588	8.69638	0.09990	-10.76840	18.98516	8.37591	30.04739	37.64733	0.00642	0.00700	4.41189	2.83353	33.99584	39.74286	40.47049	1.16753	1.91435	0.73387	0.95108	1.99388	2.30260	31.22407	24.89508	0.44697	4.18869	16.86279	