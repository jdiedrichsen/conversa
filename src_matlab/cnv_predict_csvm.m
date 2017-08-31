function predictions = cnv_predict_csvm(model, predictors)
predictions = predict(model, cnv_struct2Matrix(predictors));
end