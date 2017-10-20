function D = cnv_loadBodyData(fileName)
% Loads a tab delimited text file of motion tracking data into a struct
% Usage
%   cnvTrackingData=cnv_loadTrackingData('fileName')
% Description
%   Reads and stores data from a Kinect face-tracking data text file
%   The data is stored in a dataframe, a structure with a 2D array of fields. 
%   By Shayaan Syed Ali
%   Last updated 09-Sep-17 joern.diedrichsen 

TOP_HEADER_LN = '==========';
N_HEADER_LNS = 3;

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

% REad header of first body 
for i=1:4
    fgetl(fid); 
end; 

% Interpret header line and mae field names 
bodypart = {'waist','spine','chest','neck','head','head_tip',...
    'upperLeg_L','lowerLeg_L','foot_L','toes_L',... 
    'upperLeg_R','lowerLeg_R','foot_R','toes_R',... 
    'collar_L','upperArm_L','foreArm_L','hand_L',...
    'collar_R','upperArm_R','foreArm_R','hand_R',...
    'thumb_L_0','thumb_L_1','thumb_L_2','thumb_L_3',...
    'index_L_0','index_L_1','index_L_2','index_L_3',...
    'middle_L_0','middle_L_1','middle_L_2','middle_L_3',...
    'ring_L_0','ring_L_1','ring_L_2','ring_L_3',...
    'pinky_L_0','pinky_L_1','pinky_L_2','pinky_L_3',...
    'thumb_R_0','thumb_R_1','thumb_R_2','thumb_R_3',...
    'index_R_0','index_R_1','index_R_2','index_R_3',...
    'middle_R_0','middle_R_1','middle_R_2','middle_R_3',...
    'ring_R_0','ring_R_1','ring_R_2','ring_R_3',...
    'pinky_R_0','pinky_R_1','pinky_R_2','pinky_R_3'}; 
    
valpart = {'conf','tx','ty','tz','rx','ry','rz'}; 

Header=fgetl(fid);
H={};
Head=Header;
currentBodypart=[]; 
while length(Head)>0
    [r,Head]=strtok(Head);
    if (~isempty(r))
        if any(strcmp(r,bodypart))
            currentBodypart=r; 
        elseif any(strcmp(r,valpart));
            H{end+1}=[currentBodypart '_' r]; 
        else 
            H{end+1}=r;
        end;
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
    D.(lower(H{i})) = A(:,i); 
end; 

% End of file is reached
