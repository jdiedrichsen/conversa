function model = cnv_learn_clinear(predictors,labels,varargin)
% Logistic regression, using specific fields
fields = {'smile_l','smile_r'}; 
Learner = 'svm'; 
Regularization='lasso'; 
Lambda  = exp(-2);

vararginoptions(varargin); 
for i=1:length(fields) 
    X(:,i)=predictors.(fields{i}); 
end;
[model.svm,FitInfo] = fitclinear(X,labels,'Regularization',Regularization,'Learner',Learner,'Lambda',Lambda);
model.fields=fields; 