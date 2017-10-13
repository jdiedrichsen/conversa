function [len,F,B,x,Fs] = cnv_checkLength(dataID);
% T = cnv_makeLabelFromAvi(name);
if (nargin<2); 
    rootDir = pwd; 
end; 
aviFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_Color.avi',dataID)); 
wavFile=fullfile(rootDir,'RawData',dataID,sprintf('%s.wav',dataID)); 
<<<<<<< Updated upstream
trackingFile=fullfile(rootDir,'RawData',dataID,sprintf('%s.txt',dataID)); 
fprintf('loading %s\n',trackingFile);
T=cnv_loadTrackingData(trackingFile);
% Add trackingData fields after they start tracking (after first point of
% change)
start = cnv_firstChangeI(T, 'exclude', {'timestamp', 'istracked', 'bodyid'});
fprintf('removed %d frames\n',start);
tracklength = T.timestamp(end)-T.timestamp(start); 
T=getrow(T,start:length(T.timestamp)); 
T.timestamp=T.timestamp-T.timestamp(1); 
=======
faceFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_Face.txt',dataID)); 
bodyFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_Body.txt',dataID)); 

F=cnv_loadFaceData(faceFile);
start = cnv_firstChangeI(F, 'exclude', {'timestamp', 'istracked', 'bodyid'});
fprintf('removed %d frames\n',start);
facelength = F.timestamp(end)-F.timestamp(start); 

B=cnv_loadFaceData(bodyFile);
start = cnv_firstChangeI(B, 'exclude', {'timestamp', 'istracked', 'bodyid'});
fprintf('removed %d frames\n',start);
bodylength = B.timestamp(end)-B.timestamp(start); 

>>>>>>> Stashed changes
A=aviinfo(aviFile); 
[x,Fs]=audioread(wavFile);
fprintf('%s: F:%2.2f B:%2.2f V:%2.2f A:%2.2f\n',dataID,facelength,bodylength,A.NumFrames./A.FramesPerSecond,length(x)/Fs);
len = [facelength,bodylength,A.NumFrames./A.FramesPerSecond,length(x)/Fs]; 
