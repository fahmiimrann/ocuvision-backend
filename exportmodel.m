%% 1. Setup and Load Model
% Ensure the model is in the workspace
if ~exist('trainedModel', 'var')
    load('Bestest.mat'); 
end

%% 2. Load and Prep Test Files
% (Your existing import code)
T_DR = readtable('TestData_DR_Cleaned - 99.csv');
T_AMD = readtable('TestData_AMD_Cleaned - 99.csv');
T_Normal = readtable('TestData_Normal_Cleaned - 99.csv');
T_Glaucoma = readtable('TestData_Glaucoma_Cleaned - 99.csv');

% Add Labels
T_DR.Class = repmat("DR", height(T_DR), 1);
T_AMD.Class = repmat("AMD", height(T_AMD), 1);
T_Normal.Class = repmat("Normal", height(T_Normal), 1);
T_Glaucoma.Class = repmat("Glaucoma", height(T_Glaucoma), 1);

% Combine
TestSet = [T_DR; T_AMD; T_Normal; T_Glaucoma];
TestSet.Class = categorical(TestSet.Class); 

%% 3. Prediction & Evaluation
% Use the predictFcn from your exported model
[predictedLabels, scores] = trainedModel.predictFcn(TestSet);

actualLabels = TestSet.Class;
testAccuracy = sum(predictedLabels == actualLabels) / numel(actualLabels);

fprintf('Final Test Accuracy: %.2f%%\n', testAccuracy * 100);

%% 4. Visualization
figure;
confusionchart(actualLabels, predictedLabels, ...
    'Title', 'Confusion Matrix: Testing Phase', ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');