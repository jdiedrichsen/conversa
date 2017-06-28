function predictions = cnv_predict_csvm(model, predictors)
% TODO: Add varargin optionArgs
predictions = predict(model, cnv_struct2Matrix(predictors));  % Observations as colummns is faster according to doc
end