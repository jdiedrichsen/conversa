function model = cnv_learn_randomGuess(predictors, labels)
% Sets up a random guessing model which guess within the range in the
% labels
model.maxGuess = max(labels);
model.minGuess = min(labels);
end