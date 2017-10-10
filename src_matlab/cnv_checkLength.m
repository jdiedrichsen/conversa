function len = cnv_checkLength(dataID);
% T = cnv_makeLabelFromAvi(name);
if (nargin<2); 
    rootDir = pwd; 
end; 
aviFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_InfraRed.avi',dataID)); 
wavFile=fullfile(rootDir,'RawData',dataID,sprintf('%s.wav',dataID)); 
trackingFile=fullfile(rootDir,'RawData',dataID,sprintf('%s.txt',dataID)); 
T=cnv_loadTrackingData(trackingFile);
% Add trackingData fields after they start tracking (after first point of
% change)
start = cnv_firstChangeI(T, 'exclude', {'timestamp', 'istracked', 'bodyid'});
fprintf('removed %d frames\n',start);
tracklength = T.timestamp(end)-T.timestamp(start); 
A=aviinfo(aviFile); 
[x,Fs]=audioread(wavFile);
fprintf('%s: T:%2.2f V:%2.2f A:%2.2f\n',dataID,tracklength,A.NumFrames./A.FramesPerSecond,length(x)/Fs);
len = [tracklength,A.NumFrames./A.FramesPerSecond,length(x)/Fs]; 
