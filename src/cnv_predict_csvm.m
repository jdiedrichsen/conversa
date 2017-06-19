function predictions = cnv_predict_csvm(model, predictors)

predictions = predict(model, predictors);  % Observations as colummns is faster according to doc

end