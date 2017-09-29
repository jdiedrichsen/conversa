function [D,T,L] = cnv_loadData(dataID,rootDir); 
% function [D,T,L] = cnv_loadData(dataID,rootDir); 
% Shortcut function to get tracking and label file 
if (nargin<3); 
    rootDir = pwd; 
end; 
trackingFile=fullfile(rootDir,'RawData',dataID,sprintf('%s.txt',dataID)); 
labelFile=fullfile(rootDir,'VideoCoding',sprintf('%s.dat',dataID)); 
T=cnv_loadTrackingData(trackingFile);
L=cnv_loadLabelFile(labelFile);
D = cnv_applyLabels(T,L);
