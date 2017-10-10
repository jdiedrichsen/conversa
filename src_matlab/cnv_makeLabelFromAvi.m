function T = cnv_makeLabelFromAvi(dataID);
% T = cnv_makeLabelFromAvi(name);
if (nargin<2); 
    rootDir = pwd; 
end; 
aviFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_Color.avi',dataID)); 
labelFile=fullfile(rootDir,'VideoCoding',sprintf('%s.dat',dataID)); 
D.pid =1; 
D.cam =1; 
D.int =1; 

A=aviinfo('aviFile'); 

D.min =NaN;
D.sec = NaN; 
D.frame =NaN; 
D.absoluteframe =A.numFrames; 
D.smile =0; 
