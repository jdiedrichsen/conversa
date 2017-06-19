function predictions = cnv_predict_clinear(model, predictors)
% TODO: Add varargin optionArgs
predictions = predict(model, predictors);  % Observations as colummns is faster according to doc
end