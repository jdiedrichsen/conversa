function model = cnv_learn_fitcsvm(predictors, labels)
% TODO: Add varargin optionArgs
model = fitcsvm(predictors,labels,'Standardize',true); % Observations as colummns is faster according to doc
end