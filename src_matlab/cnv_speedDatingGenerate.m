function varargout=cnv_speedDatingGenerate(session,numMale,numFemale,numStations); 
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
participantNum(participantNum>0)=participantNum(participantNum>0)+1000*session;   
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

T.station(T.participant==0 | T.partner==0,1)=0; % Make all empty interaction to rest station 
T.camera(T.participant==0 | T.partner==0,1)=0; % Make all empty interaction to rest station 


% Check if any stattions are faulty - if yes, repeat those interactions in
% different stations. 
if (numStations<numPairs) 
    faulty = [numStations+1:numPairs]; 
    i=find(ismember(T.station,faulty)); 
    A=getrow(T,i); 
    T.station(i)=0; 
    T.camera(i)=0; 
    T.partner(i)=0; 

    int = max(T.interaction)+1; 
    while (~isempty(A.participant))
        sn=unique(A.participant);
        for s=1:min(length(sn),numStations); 
            j=find(A.participant==sn(s)); 
            if(~isempty(j)) 
                j=j(1);
                % Find complementary index 
                k= find(A.participant==A.partner(j) & A.partner == A.participant(j)); 
                A.station([j k])=s; 
                A.camera([j k])=A.station([j k])*2-mod(A.camera([j k]),2); 
                A.interaction([j k]) = int; 
                indx = [1:length(A.participant)]; 
                indx([j k])=[]; 
                T=addstruct(T,getrow(A,[j k])); 
                A=getrow(A,indx);
            end; 
        end; 
        int=int+1; 
    end; 
end; 
numInteractions = max(T.interaction); 


SS = cell(numStations+1,1); 
for s=0:numStations 
    for i=1:numInteractions 
        ind=find(T.station==s & T.interaction==i & T.participant>0); 
        for j=1:length(ind) 
            S.station     =s; 
            S.interaction =i; 
            S.camera = T.camera(ind(j)); 
            S.participant= T.participant(ind(j)); 
            % Find the next one 
            nextind=find(T.interaction==i+1 & T.participant==S.participant); 
            if isempty(nextind) 
                S.nextStation = 0; 
            else 
                S.nextStation = T.station(nextind(1)); 
            end; 
            SS{s+1}=addstruct(SS{s+1},S); 
        end; 
    end; 
end; 

dsave(sprintf('Master_%d.txt',session),T); 
for i=1:numStations+1 
    if (~isempty(SS{i}))
        dsave(sprintf('Session_%d_Station_%2.2d.txt',session,i-1),SS{i}); 
    end;
end; 
    
varargout={T,SS}; 