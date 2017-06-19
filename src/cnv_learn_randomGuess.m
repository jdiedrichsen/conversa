function model = cnv_learn_randomGuess(predictors, labels)
% Sets up a random guessing model which guesses within the range of labels
% By Shayaan Syed Ali
% Last updated 19-Jun-17
model.maxGuess = max(labels);
model.minGuess = min(labels);
end