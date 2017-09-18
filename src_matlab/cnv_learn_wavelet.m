function model = cnv_learn_wavelet(predictors,labels,varargin)
% Logistic regression, using specific fields
fields = {'jaw_open'};
wname = 'morl'; 
scales = 3:1:32; 
Learner = 'logistic'; 
Regularization='lasso'; 
Lambda  = exp(-7);
showPlot = 1; 

vararginoptions(varargin); 
Fs = 1./(predictors.timestamp(2)-predictors.timestamp(1)); 

for i=1:length(fields) 
    B = cwt(predictors.(fields{i}),scales,wname);
    freq = scal2frq(scales,wname,1/Fs);
    if (showPlot) 
        subplot(3,1,[1 2]); 
        imagesc(predictors.timestamp,freq,abs(B));
        axis tight; xlabel('Seconds'); ylabel('Pseudo-Frequency (Hz)');
        colormap(jet); 
        subplot(3,1,3); 
        imagesc(predictors.timestamp,1,labels');
        keyboard; 
    end; 
end; 
    [model.svm,FitInfo] = fitclinear(X,labels,'Regularization',Regularization,'Learner',Learner,'Lambda',Lambda);
model.fields=fields; 