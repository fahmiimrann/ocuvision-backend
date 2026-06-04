% inspect_model.m
% ------------------------------------------------------------------
% Run this from MATLAB (IDE or command window) FROM THIS FOLDER:
%
%     >> cd 'C:\Users\USER\Desktop\Fahmi - BACH\...\html\ocuvision'
%     >> inspect_model
%
% It loads 99_BaggedTreesModel.mat, prints:
%   * the exact feature names the model expects (RequiredVariables)
%   * the min / mean / max of every feature, PER class (DR, AMD,
%     Normal, Glaucoma), as the model saw them during training.
%
% Copy the entire console output and paste it back into Cursor so I
% can compare it against the live values our Node runner extracts
% from your uploaded fundus image, and patch the discrepancy.
% ------------------------------------------------------------------

clc;
fprintf('=== OcuVision model inspection ===\n');

%% 1. Load the model
modelFile = '99_BaggedTreesModel.mat';
if ~exist(modelFile, 'file')
    error('Could not find %s in the current folder. cd into the ocuvision folder first.', modelFile);
end
S = load(modelFile);
fprintf('Top-level variables in the .mat file:\n');
disp(fieldnames(S));

% Find the variable that owns predictFcn
modelVar = [];
modelName = '';
keys = fieldnames(S);
for k = 1:numel(keys)
    v = S.(keys{k});
    if isstruct(v) && isfield(v, 'predictFcn')
        modelVar = v;
        modelName = keys{k};
        break;
    end
end
if isempty(modelVar)
    error('No struct with a predictFcn field was found inside the .mat file.');
end
fprintf('Using model variable: %s\n\n', modelName);

%% 2. Expected feature names
if isfield(modelVar, 'RequiredVariables')
    req = cellstr(modelVar.RequiredVariables);
    fprintf('--- RequiredVariables (%d features) ---\n', numel(req));
    for i = 1:numel(req)
        fprintf('  %2d. %s\n', i, req{i});
    end
    fprintf('\n');
else
    req = {};
    fprintf('Model has no RequiredVariables field.\n');
end

%% 3. Training-data statistics, per class
ens = [];
if isfield(modelVar, 'ClassificationEnsemble')
    ens = modelVar.ClassificationEnsemble;
elseif isfield(modelVar, 'ClassificationSVM')
    ens = modelVar.ClassificationSVM;
elseif isfield(modelVar, 'ClassificationTree')
    ens = modelVar.ClassificationTree;
elseif isfield(modelVar, 'ClassificationDiscriminant')
    ens = modelVar.ClassificationDiscriminant;
end

if isempty(ens)
    fprintf('Could not find a ClassificationEnsemble inside the model.\n');
else
    try
        Xraw = ens.X;
        Y    = ens.Y;
        names = ens.PredictorNames;

        % Convert the predictor matrix to plain double, regardless of whether
        % MATLAB stored it as a table, dataset, or matrix.
        if istable(Xraw)
            Xt = Xraw;
            X  = table2array(Xt);
        else
            X  = Xraw;
        end
        X = double(X);

        classes = categories(categorical(Y));
        fprintf('--- Training-data feature ranges (min / mean / max) per class ---\n');
        fprintf('Total training rows: %d, features: %d, classes: %s\n\n', ...
            size(X,1), size(X,2), strjoin(string(classes), ', '));

        for c = 1:numel(classes)
            cls = classes{c};
            mask = (string(Y) == string(cls));
            Xc = X(mask, :);
            fprintf('=== Class: %s (%d samples) ===\n', cls, sum(mask));
            for f = 1:numel(names)
                fprintf('  %-22s  min=%12.6g  mean=%12.6g  max=%12.6g\n', ...
                    names{f}, min(Xc(:,f)), mean(Xc(:,f)), max(Xc(:,f)));
            end
            fprintf('\n');
        end
    catch err
        fprintf('Could not extract training data: %s\n', err.message);
        rethrow(err);
    end
end

%% 4. One worked example per class (first training row of each class)
if ~isempty(ens) && exist('X','var')
    try
        fprintf('--- One worked example row per class (first training sample) ---\n');
        for c = 1:numel(classes)
            cls = classes{c};
            idx = find(string(Y) == string(cls), 1, 'first');
            if isempty(idx); continue; end
            row = X(idx, :);
            fprintf('Class %s:\n', cls);
            for f = 1:numel(names)
                fprintf('  %-22s = %g\n', names{f}, row(f));
            end
            fprintf('\n');
        end
    catch err
        fprintf('Could not print example rows: %s\n', err.message);
    end
end

fprintf('=== End of inspection ===\n');
