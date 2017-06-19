function predictions = cnv_predict_fitcsvm(model, predictors)

predictions = predict(model, predictors);  % Observations as colummns is faster according to doc

end