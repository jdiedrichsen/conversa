function model = cnv_learn_clinear(predictors, labels)
model = fitclinear(cnv_struct2Matrix(predictors), cnv_struct2Matrix(labels));
end