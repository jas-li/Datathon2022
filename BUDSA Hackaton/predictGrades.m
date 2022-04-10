clear all;
clc;

trainSet = readtable("train.csv");
testSet = readtable("test.csv");

grades = trainSet.grade; %grades from training set
count = 1;

table_size = size(trainSet); 
cols = [8, 9, 15, 16, 27, 28, 29, 30, 31]; 
for col = cols 
    test = table2array(trainSet(:,col));
%     disp(col)
%     disp(corrcoef(test,grades)); see the correlation coefficients of each
%     column to grades
    data(:, count) = test; %append the selected columns into data matrix
    count = count + 1;
end

sizeSet = size(data);
data(:,sizeSet(2)+1) = grades; %adding in the grades column

[trainedModel RMSE] = trainRegressionModel(data);

count = 1; %making prediction inputs
for col = cols 
    test = table2array(testSet(:,col));
    predictionData(:, count) = test;
    count = count + 1;
end

yfit = trainedModel.predictFcn(predictionData);
grade = round(yfit);
ID = testSet.ID;

modelSet = table(ID, grade);

writetable(modelSet, 'submission.csv')