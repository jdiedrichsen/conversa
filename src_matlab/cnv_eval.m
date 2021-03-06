function T = cnv_eval(varargin);
% function T = cnv_eval(pid,cam,targetFields,algorithms,varargin);
% Evaluates learning algorithms on a set of learning algorithms
% Returns a crossvalidated accuracy
% INPUT (VARARGIN):
%       'pid': Vector of PIDs to consider
%       'cam': Vector (same lenght) of cameras
%       'targetField': target field ('smile','talk',etc)
%       'algorithms': Cell array of algorithms
%           Each algorithm <alg> name must be accompanied by a
%           Matlab script with the prefix cnv_learn_<alg> and
%           cnv_predict_<alg> for learning and prediction
%       'plot',{fields}: For each test set, plot set of predictors, truth and
%           prediction. You need to give ot some variables to plot
% VARARGIN:
%       'crossval',
%               'within': 5-way split within each person
%               'between': between person (leave-one-out) classification
%
% OUTPUT:
%       T: Data frame of evalation
%
% By joern Diedrichsen & Shayaan Syed Ali

pid  =[1001;1005;1005;2001;2001];
cam  =[2;1;2;1;2;1;2];

nPart = 5;            % Number of partitions for within-person classification
crossval = 'within';  % Between is between person classification
lossfcn  = 'error';    % Loss function
verbose  = 1;
targetField = 'smile';
allN = {'neckposx','neckposy','neckposz','neckrotx','neckroty','neckrotz'};
allH = {'headposx','headposy','headposz','headrotx','headroty','headrotz'};
allE = {'brow_up_l','brow_up_r','brow_down_l','brow_down_r','eye_closed_l','eye_closed_r'};
allM = {'cheek_puffed_l','cheek_puffed_r','lips_pucker','lips_stretch_l','lips_stretch_r','lip_lower_down_l','lip_lower_down_r'};
allX = {'smile_l','smile_r','frown_l','frown_r'};
allJ = {'jaw_l','jaw_r','jaw_open'};

algorithms = {'clinear','clinear','null'};
options  = {{'fields',horzcat(allJ),'Learner','logistic','Lambda',exp(-7)},...
    {'fields',horzcat(allM,allJ,allE),'Learner','logistic','Lambda',exp(-7)},...
    {}};         % Additional options for algorithm
plotPred = {};
vararginoptions(varargin);

nAlgos = length(algorithms);
if (length(options)<nAlgos)
    options{nAlgos}={};
end;

% Get Data
if (verbose)
    fprintf('Loading data...\n');
end;
nData = length(pid);
Data =[];
for i=1:nData
    dataID = sprintf('p%dcam%d',pid(i),cam(i)); 
    D=cnv_loadData(dataID);
    v= ones(length(D.timestamp),1);
    D.dataset = v*i;
    D.pid = v*pid(i);
    D.cam = v*cam(i);
    Data = addstruct(Data,D);
end;

% Set learning and prediction function handels
if ischar(algorithms)
    algorithms={algorithms};
end;
for i = 1:nAlgos
    % Set learning function
    learnFcn{i} = strcat('cnv_learn_', algorithms{i});
    predictFcn{i} = strcat('cnv_predict_', algorithms{i});
end;

% Now do crossvalidation in two different ways
T=[];
D=[];
switch (crossval)
    case 'between'
        % Remove the targetField from the data to prevent cheating!
        Labels = Data.(targetField);
        Data = rmfield(Data,targetField);
        
        % Now leave one data set out at a time
        for i=1:nData
            trainIndx = Data.dataset~=i;
            testIndx  = Data.dataset==i;
            prediction = zeros(sum(testIndx),nAlgos);
            for m=1:nAlgos
                if (verbose)
                    fprintf('Data set %d %d, algorithm %s\n',pid(i),cam(i),algorithms{m});
                end;
                D.dataset = i;
                D.algorithmNum = m;
                D.algorithm = algorithms(m);
                D.pid   = pid(i);
                D.cam   = cam(i);
                D.target  = {targetField};
                M = feval(learnFcn{m},getrow(Data,trainIndx),Labels(trainIndx,1),options{m}{:});
                prediction(:,m) = feval(predictFcn{m},M,getrow(Data,testIndx));
                D.meanPred = mean(prediction(:,m));
                D.propPred = mean(prediction(:,m)>0.5);
                D.propData = mean(Labels(testIndx,1));
                D.loss = evalLoss(Labels(testIndx,1),prediction(:,m),lossfcn);   % Calculate Loss
                T=addstruct(T,D);
            end;
            if (~isempty(plotPred))
                testData = getrow(Data,testIndx);
                time=Data.timestamp(testIndx,1);
                subplot(nAlgos+1,1,1);
                X=[];
                for j=1:length(plotPred)
                    X=[X testData.(plotPred{j})];
                end;
                plot(time,X);
                l=legend(plotPred);
                set(l,'interpreter','none'); % Prevents interpretation of underscores as subscripts in legends
                title('data');
                cnv_drawPatches(time,Labels(testIndx,1),'k');
                for m=1:nAlgos
                    subplot(nAlgos+1,1,1+m);
                    plot(time,prediction(:,m));
                    cnv_drawPatches(time,prediction(:,m)>0.5,'k');
                    title(sprintf('%s: %2.2f',algorithms{m},T.loss(T.algorithmNum==m &  T.dataset==i)));
                end;
                keyboard;
            end;
        end;
        
    case 'within'
        % Remove the targetField from the data to prevent cheating!
        Labels = Data.(targetField);
        Data = rmfield(Data,targetField);
        
        % Now do one data set at a time
        for i=1:nData
            iData = getrow(Data,Data.dataset==i);
            iLabel = Labels(Data.dataset==i,1);
            
            % Make the partition vector
            nFrames = length(iData.timestamp);
            iData.part = ceil([1:nFrames]'./nFrames*(nPart));
            
            % Loop over partitions and build up prediction
            prediction = zeros(nFrames,nAlgos);
            for m=1:nAlgos
                if (verbose)
                    fprintf('Data set %d %d, algorithm %s\n',pid(i),cam(i),algorithms{m});
                end;
                for j=1:nPart
                    trainIndx = iData.part~=j;
                    testIndx  = iData.part==j;
                    M = feval(learnFcn{m},getrow(iData,trainIndx),iLabel(trainIndx,1),options{m}{:});
                    prediction(testIndx,m) = feval(predictFcn{m},M,getrow(iData,testIndx));
                end;
                D.dataset = i;
                D.algorithmNum = m;
                D.algorithm = algorithms(m);
                D.pid   = pid(i);
                D.cam   = cam(i);
                D.target  = {targetField};
                D.meanPred = mean(prediction(:,m));
                D.propPred = mean(prediction(:,m)>0.5);
                D.propData = mean(iLabel);
                D.loss = evalLoss(iLabel,prediction(:,m),lossfcn);   % Calculate Loss
                T=addstruct(T,D);
            end;
        end;
end

% Recieves predicted and actual as matrices (or vectors)
function loss = evalLoss(actual,prediction,lossfcn)
switch (lossfcn)
    case 'error'
        loss = mean(abs(actual-(prediction>0.5))); % Mean prediction error
    case 'abserror'
        loss = mean(abs(actual-(prediction))); % Mean prediction error
end;