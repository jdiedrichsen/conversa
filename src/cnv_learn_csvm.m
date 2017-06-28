function model = cnv_learn_fitcsvm(predictors, labels)
model = fitcsvm(cnv_struct2Matrix(predictors),cnv_struct2Matrix(labels),'Standardize',true); % Observations as colummns is faster according to doc
end