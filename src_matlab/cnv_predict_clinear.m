function prediction = cnv_predict_clinear(model, data)
for i=1:length(model.fields) 
    X(:,i)=data.(model.fields{i}); 
end; 
prediction = predict(model.svm,X); 