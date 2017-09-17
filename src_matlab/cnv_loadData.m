function [D,T,L] = cnv_loadData(pid,cam,rootDir); 
% function [D,T,L] = cnv_loadData(pid,cam,rootDir); 
% Shortcut function to get tracking and label file 
if (nargin<3); 
    rootDir = pwd; 
end; 
trackingFile=fullfile(rootDir,'RawData',sprintf('par%dCam%d',pid,cam),...
    sprintf('cam%dpar%d.txt',cam,pid)); 
labelFile=fullfile(rootDir,'VideoCoding',sprintf('p%dcam%d.dat',pid,cam)); 
T=cnv_loadTrackingData(trackingFile); 
L=cnv_loadLabelFile(labelFile); 
D = cnv_applyLabels(T,L);
