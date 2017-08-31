function predictions = cnv_predict_clinear(model, predictors)
predictions = predict(model, cnv_struct2Matrix(predictors));
end