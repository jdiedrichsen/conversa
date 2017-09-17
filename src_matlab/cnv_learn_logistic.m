function model = cnv_learn_logistic(predictors, label,varargin)
% Logistic regression, using specific fields
fields = {'smile_l','smile_r'}; 
vararginoptions(varargin); 
for i=1:length(fields) 
    X(:,i)=predictors.(fields{i}); 
end; 
model.fields = fields; 
model.B = glmfit(X,label,'binomial','link','logit'); 