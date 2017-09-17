function model = cnv_learn_null(predictors, label,varargin)
% Null model: learns only mean probability 
model.mean = mean(label); 