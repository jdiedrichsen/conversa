function prediction = cnv_predict_null(model, predictors)
predictFields = fieldnames(predictors);
prediction = zeros(length(predictors.(predictFields{1})),model.nLabelFields);
end