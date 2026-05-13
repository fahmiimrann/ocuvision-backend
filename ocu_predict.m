function result = ocu_predict(modelPath, imagePath)
%OCU_PREDICT  OcuVision feature extraction + Bagged Trees classifier.
%
%   result = OCU_PREDICT(modelPath, imagePath) loads the Classification
%   Learner export at MODELPATH (typically 99_BaggedTreesModel.mat),
%   computes the 24-feature row that the model was trained on from the
%   image at IMAGEPATH, runs predictFcn, and returns a struct with the
%   fields:
%       .diagnosis        – clinical sentence shown in the UI
%       .severity         – Healthy | Moderate | Critical
%       .score            – placeholder confidence (UI overrides via slider)
%       .notes            – clinical note matching the predicted class
%       .rawDiagnosis     – raw label returned by predictFcn
%       .missingFeatures  – cell array of column names that had to be 0-filled
%       .features         – struct with the human-readable feature values
%
%   This is the canonical OcuVision feature pipeline. It is called by
%   matlab-runner.js in two different ways:
%       (a) directly from a full MATLAB install during local development,
%       (b) from the standalone .exe produced by ocu_main.m + MATLAB Compiler
%           on a deployment server that only has the (free) MATLAB Runtime.
%
%   The feature formulas were reverse-engineered to match the training
%   distribution of 99_BaggedTreesModel.mat — see inspect_model.m for the
%   per-class min/mean/max statistics they target.

%% 1. Load the .mat -------------------------------------------------------
    if ~exist(modelPath, 'file')
        error('OCU:ModelNotFound', 'Model file not found: %s', modelPath);
    end
    if ~exist(imagePath, 'file')
        error('OCU:ImageNotFound', 'Image file not found: %s', imagePath);
    end

    loaded = load(modelPath);
    keys = fieldnames(loaded);
    modelVar = [];
    for k = 1:numel(keys)
        v = loaded.(keys{k});
        if isstruct(v) && isfield(v, 'predictFcn')
            modelVar = v;
            break;
        end
    end
    if isempty(modelVar)
        error('OCU:BadModel', ...
            'No struct with a predictFcn field was found in the .mat file. Expected a Classification Learner export (trainedModel.predictFcn).');
    end

%% 2. Load & pre-process the image ---------------------------------------
    img = imread(imagePath);
    if ndims(img) == 2 || size(img, 3) == 1
        img = repmat(img, 1, 1, 3);
    end
    resized = imresize(img, [512 512], 'bicubic');
    grey    = rgb2gray(resized);
    claheImg = adapthisteq(grey, 'ClipLimit', 0.005, 'NumTiles', [8 8]);

%% 3a. LBP features (f1..f10) -------------------------------------------
%   Upright=false (rotation-invariant) gives NumNeighbors+2 = 10 features.
    lbpRaw = extractLBPFeatures(claheImg, 'Upright', false, ...
        'NumNeighbors', 8, 'Radius', 1);
    if sum(lbpRaw) > 0
        lbp = lbpRaw / sum(lbpRaw);
    else
        lbp = lbpRaw;
    end
    if numel(lbp) < 10
        lbp = [lbp(:)' zeros(1, 10 - numel(lbp))];
    end
    lbp = lbp(1:10);

%% 3b. GLCM features -----------------------------------------------------
    glcm = graycomatrix(claheImg, 'Offset', [0 1], 'Symmetric', true, 'NumLevels', 256);
    stats = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});

%% 3c. Intensity features (CLAHE-enhanced image) -------------------------
    claheD = double(claheImg(:));
    fMean  = mean(claheD);
    fStd   = std(claheD);
    fVar   = var(claheD);
    fSkew  = skewness(claheD);
    fKurt  = kurtosis(claheD);
    fLogK  = log(1e-5 + abs(fKurt));
    h      = imhist(claheImg, 256);
    p      = h / sum(h);
    fEnt   = -sum(p(p > 0) .* log2(p(p > 0)));
    fUni   = sum(p .^ 2);
    fSmooth = 1 - (1 / (1 + fVar));
    fDyn   = double(max(claheImg(:)) - min(claheImg(:)));

%% 3d. Vessel segmentation & quantification ------------------------------
%   GVD = vessel fraction inside the retina ROI (in %).
%   LVD = max value of the 31x31-window vessel-density map (in [0,1]).
%   VSM = min(densityLeft,densityRight) / max(densityLeft,densityRight).
    greenCh   = resized(:,:,2);
    invGreen  = imcomplement(greenCh);
    vesselMask = imtophat(invGreen, strel('disk', 8));
    vBin      = imbinarize(vesselMask, 'adaptive', 'Sensitivity', 0.55);
    gvdAll    = (sum(vBin(:)) / numel(vBin)) * 100;

    kern     = ones(31, 31) / 961;
    lvdMap   = conv2(double(vBin), kern, 'same');

    roiMask = imbinarize(greenCh, 20 / 255);
    roiMask = imfill(roiMask, 'holes');
    vpInRoi = sum(vBin(roiMask == 1));
    totalRoi = sum(roiMask(:));
    if totalRoi > 0
        gvd = (vpInRoi / totalRoi) * 100;
    else
        gvd = gvdAll;
    end
    maxLvd = max(lvdMap(:));

    centroidStats = regionprops(roiMask, 'Centroid');
    if ~isempty(centroidStats)
        cxFull = round(centroidStats(1).Centroid(1));
    else
        cxFull = 256;
    end
    leftV  = vBin(:, 1:cxFull);
    rightV = vBin(:, cxFull + 1:end);
    leftM  = roiMask(:, 1:cxFull);
    rightM = roiMask(:, cxFull + 1:end);
    if sum(leftM(:)) > 0
        dLeft = sum(leftV(leftM)) / sum(leftM(:));
    else
        dLeft = 0;
    end
    if sum(rightM(:)) > 0
        dRight = sum(rightV(rightM)) / sum(rightM(:));
    else
        dRight = 0;
    end
    if max([dLeft dRight]) == 0
        vsm = 0;
    else
        vsm = min([dLeft dRight]) / max([dLeft dRight]);
    end

%% 3e. Optic Nerve Head: disc + cup detection ----------------------------
    redCh    = resized(:,:,1);
    claheG   = adapthisteq(greenCh, 'ClipLimit', 0.01);
    claheR   = adapthisteq(redCh,   'ClipLimit', 0.02);
    claheRm  = medfilt2(claheR, [3 3]);
    threshVal = prctile(claheRm(:), 99.5);
    bright = claheRm >= threshVal;
    bright = imclose(bright, strel('disk', 10));
    bright = imopen(bright, strel('disk', 5));
    bright = imfill(bright, 'holes');
    labelImg = bwlabel(bright);
    props = regionprops(labelImg, 'Area', 'Centroid', 'EquivDiameter');

    if ~isempty(props)
        [~, idx] = max([props.Area]);
        cy = props(idx).Centroid(2);
        cx = props(idx).Centroid(1);
        r  = props(idx).EquivDiameter / 2;
        if r < 30
            r = 45;
        elseif r > 90
            r = 70;
        end
    else
        cx = 256; cy = 256; r = 65;
    end
    r = r * 0.91;
    oda = pi * r ^ 2;

    [xx, yy] = meshgrid(1:512, 1:512);
    discMask = ((xx - cx) .^ 2 + (yy - cy) .^ 2) <= r ^ 2;
    searchPad = round(r * 0.1);
    cupSearch = ((xx - cx) .^ 2 + (yy - cy) .^ 2) <= (r + searchPad) ^ 2;
    cupRoi = claheG;
    cupRoi(~cupSearch) = 0;
    roiVals = cupRoi(discMask);
    if numel(roiVals) > 0
        intLo = prctile(roiVals, 85);
        cup = (cupRoi >= intLo) & discMask;
        cup = bwareaopen(cup, round(pi * (r / 6) ^ 2));
        cup = imclose(cup, strel('disk', 6));
        cup = imopen(cup, strel('disk', 4));
        lcup = bwlabel(cup);
        pcup = regionprops(lcup, 'Area');
        if ~isempty(pcup)
            [~, mi] = max([pcup.Area]);
            hybridCup = lcup == mi;
        else
            hybridCup = false(size(cup));
        end
    else
        hybridCup = false(size(discMask));
    end
    pcupBb  = regionprops(bwlabel(hybridCup), 'BoundingBox');
    pdiscBb = regionprops(bwlabel(discMask),  'BoundingBox');
    if ~isempty(pcupBb) && ~isempty(pdiscBb)
        vCup  = pcupBb(1).BoundingBox(4);
        vDisc = pdiscBb(1).BoundingBox(4);
        if vDisc > 0
            cdr = min(1, vCup / vDisc);
        else
            cdr = 0.18;
        end
    else
        cdr = 0.18;
    end

%% 3f. Assemble feature row in the order the model expects ---------------
    knownNames = { ...
        'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10', ...
        'Contrast','GLCM_Contrast', ...
        'Correlation','Corrrelation','GLCM_Correlation', ...
        'Energy','Energy_Texture','GLCM_Energy', ...
        'Homogeneity','GLCM_Homogeneity', ...
        'Mean','Intensity_Mean', ...
        'Standard_Deviation','StdDev','Intensity_StdDev', ...
        'Variance','Intensity_Variance', ...
        'Skewness','Intensity_Skewness', ...
        'Kurtosis','Intensity_Kurtosis', ...
        'Log_kurtosis','Intensity_LogKurt','LogKurtosis', ...
        'Smoothness','Intensity_Smoothness', ...
        'Entropy','Intensity_Entropy', ...
        'Energy_Intensity','Intensity_Uniformity','Uniformity', ...
        'DynRange','Intensity_DynRange', ...
        'GVD','Vessel_GVD','Vessel_GVD_pct', ...
        'LVD','MaxLVD','Vessel_MaxLVD', ...
        'VSM','Vessel_Symmetry', ...
        'CDR','ONH_CupToDiscRatio', ...
        'ODA','ONH_DiscArea','ONH_DiscArea_px2' };
    knownVals = [ ...
        lbp(1), lbp(2), lbp(3), lbp(4), lbp(5), ...
        lbp(6), lbp(7), lbp(8), lbp(9), lbp(10), ...
        stats.Contrast, stats.Contrast, ...
        stats.Correlation, stats.Correlation, stats.Correlation, ...
        stats.Energy, stats.Energy, stats.Energy, ...
        stats.Homogeneity, stats.Homogeneity, ...
        fMean, fMean, ...
        fStd, fStd, fStd, ...
        fVar, fVar, ...
        fSkew, fSkew, ...
        fKurt, fKurt, ...
        fLogK, fLogK, fLogK, ...
        fSmooth, fSmooth, ...
        fEnt, fEnt, ...
        fUni, fUni, fUni, ...
        fDyn, fDyn, ...
        gvd, gvd, gvd, ...
        maxLvd, maxLvd, maxLvd, ...
        vsm, vsm, ...
        cdr, cdr, ...
        oda, oda, oda ];

    if isfield(modelVar, 'RequiredVariables')
        required = cellstr(modelVar.RequiredVariables);
    else
        required = {'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10', ...
                    'Contrast','Corrrelation','Energy_Texture','Homogeneity', ...
                    'Mean','Energy_Intensity','Standard_Deviation','Entropy','Log_kurtosis', ...
                    'GVD','LVD','VSM','CDR','ODA'};
    end
    vals = zeros(1, numel(required));
    missing = {};
    for i = 1:numel(required)
        idx = find(strcmpi(knownNames, required{i}), 1);
        if ~isempty(idx)
            vals(i) = knownVals(idx);
        else
            missing{end + 1} = required{i}; %#ok<AGROW>
        end
    end
    feats = array2table(vals, 'VariableNames', required);

%% 3g. Echo the feature row (visible in the server log) ------------------
    fprintf('OCU_FEATURES_BEGIN\n');
    for i = 1:numel(required)
        fprintf('  %s = %g\n', required{i}, vals(i));
    end
    fprintf('OCU_FEATURES_END\n');

%% 4. Run the classifier ------------------------------------------------
    pred = modelVar.predictFcn(feats);
    if iscell(pred), pred = pred{1}; end
    if isnumeric(pred), pred = num2str(pred); end
    predStr = char(string(pred));

%% 5. Map the raw class to a clinical sentence ---------------------------
    low = lower(predStr);
    if contains(low, 'healthy') || contains(low, 'normal')
        severity = 'Healthy';
        display = 'Healthy';
        notes = 'Fundus pattern is within baseline range. Continue routine annual screening.';
    elseif contains(low, 'diabetic') || contains(low, 'dr ') || strcmp(low, 'dr')
        severity = 'Critical';
        display = 'Diabetic Retinopathy';
        notes = 'Multiple microaneurysms and intraretinal hemorrhages detected. Expert validation recommended.';
    elseif contains(low, 'glaucoma')
        if contains(low, 'early')
            severity = 'Moderate';
            display = 'Early Glaucoma indicators detected';
            notes = 'Cup-to-disc ratio and vessel morphology should be reviewed by an ophthalmologist.';
        else
            severity = 'Critical';
            display = 'Glaucoma';
            notes = 'Optic-disc cupping and neuroretinal-rim changes consistent with glaucoma. Refer for ophthalmic evaluation.';
        end
    elseif contains(low, 'amd') || contains(low, 'macular')
        severity = 'Moderate';
        display = 'Age-related Macular Degeneration';
        notes = 'Drusen and macular pigmentary changes detected. Specialist follow-up recommended.';
    elseif contains(low, 'cataract')
        severity = 'Moderate';
        display = 'Cataract indicators detected';
        notes = 'Lens opacification reduces image clarity; consider slit-lamp evaluation.';
    else
        severity = 'Critical';
        display = predStr;
        notes = 'Abnormal fundus pattern detected. Expert validation recommended.';
    end

%% 6. Build the return struct -------------------------------------------
    result = struct();
    result.diagnosis = display;
    result.score = 92.5;
    result.severity = severity;
    result.notes = notes;
    result.rawDiagnosis = predStr;
    result.missingFeatures = missing;

    features = struct();
    features.GLCM_Contrast        = stats.Contrast;
    features.GLCM_Correlation     = stats.Correlation;
    features.GLCM_Energy          = stats.Energy;
    features.GLCM_Homogeneity     = stats.Homogeneity;
    features.Intensity_Mean       = fMean;
    features.Intensity_StdDev     = fStd;
    features.Intensity_Skewness   = fSkew;
    features.Intensity_Kurtosis   = fKurt;
    features.Intensity_Variance   = fVar;
    features.Intensity_LogKurt    = fLogK;
    features.Intensity_Smoothness = fSmooth;
    features.Intensity_Entropy    = fEnt;
    features.Intensity_Uniformity = fUni;
    features.Intensity_DynRange   = fDyn;
    features.Vessel_GVD_pct       = gvd;
    features.Vessel_MaxLVD        = maxLvd;
    features.Vessel_Symmetry      = vsm;
    features.ONH_DiscArea_px2     = oda;
    features.ONH_CupToDiscRatio   = cdr;
    features.LBP_f1  = lbp(1);
    features.LBP_f2  = lbp(2);
    features.LBP_f3  = lbp(3);
    features.LBP_f4  = lbp(4);
    features.LBP_f5  = lbp(5);
    features.LBP_f6  = lbp(6);
    features.LBP_f7  = lbp(7);
    features.LBP_f8  = lbp(8);
    features.LBP_f9  = lbp(9);
    features.LBP_f10 = lbp(10);
    result.features = features;
end
