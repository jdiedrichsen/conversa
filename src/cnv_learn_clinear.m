function model = cnv_learn_clinear(predictors, labels)
% TODO: Add varargin optionArgs
model = fitclinear(predictors, labels);
end