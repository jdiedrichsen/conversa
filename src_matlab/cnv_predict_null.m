function prediction = cnv_predict_null(model, data)
% Predicts the mean probability 
prediction = model.mean*ones(length(data.timestamp),1);
