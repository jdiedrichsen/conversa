function [len,F,B,x,Fs] = cnv_checkLength(dataID);
% T = cnv_makeLabelFromAvi(name);
if (nargin<2); 
    rootDir = pwd; 
end; 
aviFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_Color.avi',dataID)); 
wavFile=fullfile(rootDir,'RawData',dataID,sprintf('%s.wav',dataID)); 
faceFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_face.txt',dataID)); 
bodyFile=fullfile(rootDir,'RawData',dataID,sprintf('%s_body.txt',dataID)); 

F=cnv_loadFaceData(faceFile);
start = cnv_firstChangeI(F, 'exclude', {'timestamp', 'istracked', 'bodyid'});
fprintf('removed %d frames\n',start);
facelength = F.timestamp(end)-F.timestamp(start); 
F=getrow(F,start:length(F.timestamp)); 
F.timestamp=F.timestamp-F.timestamp(1); 

B=cnv_loadBodyData(bodyFile);
start = cnv_firstChangeI(B, 'exclude', {'timestamp', 'istracked', 'bodyid'});
fprintf('removed %d frames\n',start);
bodylength = B.timestamp(end)-B.timestamp(start); 
B=getrow(B,start:length(B.timestamp)); 
B.timestamp=B.timestamp-B.timestamp(1); 

A=aviinfo(aviFile); 
[x,Fs]=audioread(wavFile);
fprintf('%s: F:%2.2f B:%2.2f V:%2.2f A:%2.2f\n',dataID,facelength,bodylength,A.NumFrames./A.FramesPerSecond,length(x)/Fs);
len = [facelength,bodylength,A.NumFrames./A.FramesPerSecond,length(x)/Fs]; 
