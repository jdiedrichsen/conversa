function [D,T,L] = cnv_loadData(dataID,rootDir); 
% function [D,T,L] = cnv_loadData(dataID,rootDir); 
% Shortcut function to get tracking and label file 
if (nargin<2); 
    rootDir = pwd; 
end; 
faceFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_face.txt',dataID)); 
labelFile=fullfile(rootDir,'VideoCoding',sprintf('%s.dat',dataID)); 
T=cnv_loadFaceData(faceFile);
L=cnv_loadLabelFile(labelFile);
D = cnv_applyLabels(T,L);
