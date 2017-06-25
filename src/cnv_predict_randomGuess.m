function prediction = cnv_predict_randomGuess(model, predictors)
% Predicts the corresponding labels given a trained model from cnv_learn_randomGuess and a matrix of predictors
% By Shayaan Syed Ali
% Last updated 19-Jun-17
nPredictions = length(predictors);
prediction = model.minGuess + (model.maxGuess-model.minGuess)*rand(nPredictions, 1);
end