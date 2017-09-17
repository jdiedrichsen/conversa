function prediction = cnv_predict_logistic(model, data)
% Predicts the mean probability 
for i=1:length(model.fields) 
    X(:,i)=data.(model.fields{i}); 
end; 
prediction = glmval(model.B,X,'logit'); 