function [T,D]=cnv_speedDatingGenerate(session,numMale,numFemale,numStations); 
% function cnv_speedDatingGenerate(session,numMale,numFemale,numStations);
% Session: Number of session 
% nuMale: Number of male participants 
% numFemale: Number of female participants 
% numStatitions: Number of working stations you have 
% 
% Sets up the Master file and the roomspecific files for 
% the speed dating study
pNumber = (session-1)*30; % Starting participant number 
numPairs = max([numMale numFemale]);
participantNum=zeros(numPairs,2);
participantNum(1:numMale,1)=pNumber+[1:2:numMale*2]; 
participantNum(1:numFemale,2)=pNumber+[2:2:numFemale*2];
interaction = zeros(numPairs,numPairs); 
T=[]; 
for i=1:numPairs
    % Generate pairing for this interaction 
    StationNum=mod([0:numPairs-1]+floor(i/2),numPairs)+1;
    if (i==1)
        I(:,:,i)=diag(StationNum(1:numPairs),0);  
    else         
        I(:,:,i)=diag(StationNum(1:numPairs-i+1),i-1)+diag(StationNum(numPairs-i+2:end),-numPairs+i-1); 
    end;
    
    for s=1:numPairs 
        % Now right participants 
        ind=find(I(:,:,i)==s); 
        [i1,i2]=ind2sub([numPairs numPairs],ind);
        
        % Build up the master sheet: 
        D.session  = [session;session];
        D.interaction = [i;i]; 
        D.station = [s;s]; 
        D.camera = [(s-1)*2+1;(s-1)*2+2];  
        if (mod(i,2)==0)
            D.participant = [participantNum(i1,1);participantNum(i2,2)];
            D.partner = [participantNum(i2,2);participantNum(i1,1)];
        else 
            D.participant = [participantNum(i2,2);participantNum(i1,1)];
            D.partner = [participantNum(i1,1);participantNum(i2,2)];
        end; 
        T=addstruct(T,D);
    end; 
end; 

% Check if any stattions are faulty - if yes, repeat those interactions in
% different stations. 
if (numStations<numPairs) 
    faulty = [numStations+1:numPairs]; 
    i=find(ismember(T.station,faulty)); 
    keyboard; 
else 
    numInteractions = numPairs; 
end; 
T.station(T.partitipant==0 | T.partner==0,1)=0; % Make all empty interaction to rest station 

for s=0:numStations 
    for i=1:numInteractions 
        ind=find(T.station==s & T.interaction==i); 
        for j=1:length(ind) 
            S{i+1}.station     =s; 
            S{i+1}.interaction =i; 
        end; 
        
        
    end; 
end; 

varargout={T}; 