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

%% 3d.b Vessel mask for visualisation (matches repair_vesselonh/final_batch.m)
%   The GVD/LVD/symmetry numbers above feed the trained Bagged Trees model
%   and were tuned against its training distribution, so we keep that mask
%   (`vBin`) untouched. For the UI we compute a second, denser binary mask:
%       1) inv_green   = imcomplement(green)
%       2) v_enhanced  = imtophat(inv_green, disk(8))
%       3) local_thresh = Gaussian-filtered v_enhanced
%       4) vessels_bin = v_enhanced > local_thresh + 5
%       5) bwareaopen(6)  — drop only the tiniest specks
%       6) imdilate(disk(1)) — 1-pixel thickening for UI visibility
%   We deliberately do NOT skeletonise: skel(...) produces anatomically
%   pretty but visually very sparse 1-px traces that vanish when the panel
%   downscales from 512x512 to ~200 px. Keeping the thick top-hat response
%   gives the dense, textured "vessel network" look the UI expects.
    vesselEnh   = imtophat(invGreen, strel('disk', 8, 0));
    hGauss      = fspecial('gaussian', [31 31], 5);
    localTh     = imfilter(double(vesselEnh), hGauss, 'replicate');
    vesselsBin0 = double(vesselEnh) > (localTh + 5);
    vesselsBin0 = vesselsBin0 & roiMask;
    vesselsBin0 = bwareaopen(vesselsBin0, 6);
    vesselsDisp = imdilate(vesselsBin0, strel('disk', 1));
    vesselsDisp = vesselsDisp & roiMask;

%% 3d.1 DR lesion detection (microaneurysms / dot-hemorrhages) -----------
%   The trained Bagged Trees model has no feature that captures DR's
%   signature pattern of small dark spots scattered across the retina,
%   so DR images often land in the Healthy cluster. We compute a
%   lesion-density score here and use it AFTER predictFcn as a safety
%   net to flip Healthy -> DR when many candidate lesions are present.
%
%   Classical computer-vision pipeline (no extra training needed):
%     1. Invert green channel  (microaneurysms become bright spots)
%     2. CLAHE                 (boost local contrast)
%     3. Top-hat morphology    (isolate small bright blobs only)
%     4. Threshold + size filter (keep blobs in the 5..200 px range,
%                                 typical for microaneurysms on 512x512)
%     5. Exclude vessel pixels (dilated vessel mask) and restrict to
%        the retina ROI to suppress false positives at the rim.
    invGreen2   = imcomplement(greenCh);
    claheInv    = adapthisteq(invGreen2, 'ClipLimit', 0.01);
    drTopHat    = imtophat(claheInv, strel('disk', 5));
    drThreshVal = prctile(drTopHat(:), 98);
    drSpots     = drTopHat > drThreshVal;
    drSpots     = drSpots & ~imdilate(vBin, strel('disk', 2));
    drSpots     = bwareaopen(drSpots, 5);
    drSpots     = drSpots & ~bwareaopen(drSpots, 200);
    drSpots     = drSpots & roiMask;
    drCC        = bwconncomp(drSpots);
    lesionCount = drCC.NumObjects;
    roiPx       = sum(roiMask(:));
    if roiPx > 0
        lesionDensity = (lesionCount / roiPx) * 1e6;  % lesions per million ROI pixels
    else
        lesionDensity = 0;
    end

%% 3e. Optic Nerve Head: disc + cup detection ----------------------------
    redCh    = resized(:,:,1);
    claheG   = adapthisteq(greenCh, 'ClipLimit', 0.01);
    claheR   = adapthisteq(redCh,   'ClipLimit', 0.02);
    claheRm  = medfilt2(claheR, [3 3]);
    % Use the 99th percentile (was 99.5) so we capture both the cup AND
    % the surrounding neuroretinal rim. On glaucomatous fundi the cup is
    % much brighter than the rim, so a 99.5 cutoff can pick up the cup
    % only, leaving the cup detector below it with nothing to compare
    % against -> falsely low CDR.
    threshVal = prctile(claheRm(:), 99);
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
    discAreaPrelim = sum(discMask(:));
    if numel(roiVals) > 0
        % Cup/rim separation via Otsu within the disc. Otsu finds the
        % NATURAL intensity break between the bright cup and the dimmer
        % neuroretinal rim regardless of how much of the disc the cup
        % occupies, unlike a fixed percentile which under-sizes the cup
        % whenever it covers more than ~15 % of the disc (typical for
        % glaucomatous fundi -> what was burying glaucoma_603's CDR).
        %
        % Sanity cap: when the disc has a near-uniform intensity
        % histogram (e.g. small/dim discs on DR fundi), Otsu's threshold
        % can drop low enough that the "cup" balloons to cover the
        % entire disc -> a spurious CDR of 1.0. If that happens we fall
        % back to a high-percentile threshold so the cup is always a
        % bounded sub-region of the disc, even on uniform discs.
        roi8 = uint8(roiVals);
        otsuLevel = graythresh(roi8);                  % normalized [0,1]
        otsuThresh = double(otsuLevel) * 255;
        pctAggr = prctile(double(roiVals), 70);
        pctSafe = prctile(double(roiVals), 85);
        cupThresh = otsuThresh;
        if ~isfinite(cupThresh) || cupThresh <= 0
            cupThresh = pctAggr;
        end
        cupTry = (cupRoi >= cupThresh) & discMask;
        cupTryArea = sum(cupTry(:));
        if discAreaPrelim > 0 && cupTryArea > 0.85 * discAreaPrelim
            % Degenerate Otsu (uniform disc): use a conservative
            % percentile so the cup stays a proper subregion. We log the
            % event so it's visible during regression debugging.
            fprintf('OCU_CUP_FALLBACK: otsu degenerate (cup=%.0f%% of disc), using prctile85.\n', ...
                    100 * cupTryArea / max(discAreaPrelim, 1));
            cupThresh = pctSafe;
        end
        fprintf('OCU_CUP_THRESH: otsu=%.2f pct70=%.2f pct85=%.2f used=%.2f\n', ...
                otsuThresh, pctAggr, pctSafe, cupThresh);
        cup = (cupRoi >= cupThresh) & discMask;
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
    pcupStats  = regionprops(bwlabel(hybridCup), 'BoundingBox', 'Centroid');
    pdiscStats = regionprops(bwlabel(discMask),  'BoundingBox', 'Centroid');
    % CDR is reported as the MAX of two independent estimators so a weak
    % cup-bbox doesn't single-handedly bury a clear glaucoma cup:
    %   * verticalCdr - bounding-box height ratio (clinical convention)
    %   * areaCdr     - cupArea/discArea, converted to a linear-equivalent
    %                   ratio via sqrt for fair comparison against vCDR
    % Both are surfaced in the features struct for the patient report.
    cupArea  = sum(hybridCup(:));
    discArea = sum(discMask(:));
    if discArea > 0
        areaCdr = cupArea / discArea;
    else
        areaCdr = 0;
    end
    if ~isempty(pcupStats) && ~isempty(pdiscStats)
        vCup  = pcupStats(1).BoundingBox(4);
        vDisc = pdiscStats(1).BoundingBox(4);
        if vDisc > 0
            verticalCdr = min(1, vCup / vDisc);
        else
            verticalCdr = 0.18;
        end
        cupCx = pcupStats(1).Centroid(1);
        cupCy = pcupStats(1).Centroid(2);
        cupRx = max(pcupStats(1).BoundingBox(3) / 2, r * 0.18);
        cupRy = max(pcupStats(1).BoundingBox(4) / 2, r * 0.18);
    else
        verticalCdr = 0.18;
        cupCx = cx;
        cupCy = cy;
        cupRx = r * 0.25;
        cupRy = r * 0.25;
    end
    % Sanity caps to prevent a degenerate cup mask (whose bounding box
    % spans the full disc even when area is small) from forcing a
    % spurious CDR of 1.0. CDR is anatomically bounded around 0.95 in
    % the most extreme glaucoma; cap at 0.85 to leave headroom and force
    % the downstream override to require corroborating evidence rather
    % than waving the result through.
    verticalCdr = min(verticalCdr, 0.85);
    areaCdr     = min(areaCdr,     0.85);
    cdr = max(verticalCdr, sqrt(max(areaCdr, 0)));
    cdr = min(cdr, 0.85);
    fprintf('OCU_ONH: discArea=%d cupArea=%d areaCdr=%.4f verticalCdr=%.4f effectiveCdr=%.4f\n', ...
            discArea, cupArea, areaCdr, verticalCdr, cdr);

    % Cup boundary polygon (matches repair_vesselonh/final_batch.m line 265).
    % We pull the largest contour from bwboundaries, downsample if dense,
    % and normalise to [0,1] in [x,y] order so the frontend can render the
    % real cup outline (not an ellipse approximation) on top of the image.
    cupContours = bwboundaries(hybridCup, 'noholes');
    if ~isempty(cupContours)
        cupBest = cupContours{1};
        for ci = 2:numel(cupContours)
            if size(cupContours{ci}, 1) > size(cupBest, 1)
                cupBest = cupContours{ci};
            end
        end
        if size(cupBest, 1) > 96
            step = max(1, floor(size(cupBest, 1) / 96));
            cupBest = cupBest(1:step:end, :);
        end
        cupBoundaryNorm = [cupBest(:,2) / 512, cupBest(:,1) / 512];
    else
        cupBoundaryNorm = zeros(0, 2);
    end

%% 3e.1 DR bright-lesion detection (hard exudates / cotton-wool spots) ---
%   Section 3d.1 catches DARK DR lesions (microaneurysms / dot-hemorrhages).
%   This section catches BRIGHT DR lesions (hard exudates, cotton-wool
%   spots) that appear yellow/white on the fundus and are very common in
%   early-to-moderate DR. Without this we miss DR images whose only signs
%   are bright lesions (we saw exactly that case during testing).
%
%   Pipeline:
%     1. CLAHE green channel  (already computed as `claheG` in 3e)
%     2. Top-hat with a medium disk (8 px) to isolate medium bright blobs.
%     3. Threshold at 97th percentile.
%     4. Restrict to retina ROI.
%     5. EXCLUDE optic disc + a 12-px halo around it (very bright, false
%        positive otherwise).
%     6. Size filter (5..500 px) — typical exudate cluster range.
    brightTopHat   = imtophat(claheG, strel('disk', 8));
    brightThresh   = prctile(brightTopHat(:), 97);
    brightSpots    = brightTopHat > brightThresh;
    brightSpots    = brightSpots & roiMask;
    discHalo       = imdilate(discMask, strel('disk', 12));
    brightSpots    = brightSpots & ~discHalo;
    brightSpots    = bwareaopen(brightSpots, 5);
    brightSpots    = brightSpots & ~bwareaopen(brightSpots, 500);
    brightCC       = bwconncomp(brightSpots);
    brightLesionCount = brightCC.NumObjects;

    darkLesionCount   = lesionCount;            % from section 3d.1
    totalLesionCount  = darkLesionCount + brightLesionCount;
    if roiPx > 0
        totalLesionDensity = (totalLesionCount / roiPx) * 1e6;
    else
        totalLesionDensity = 0;
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

    % Compute REAL classifier confidence from class probabilities.
    % `predictFcn` only returns the winning label, but the underlying
    % Classification Learner ensemble can return the full score vector
    % via predict(). We probe every supported ensemble type so this
    % works for Bagged Trees, SVM, KNN, Tree, and Discriminant exports.
    classConfidence = NaN;
    ens = [];
    if isfield(modelVar, 'ClassificationEnsemble')
        ens = modelVar.ClassificationEnsemble;
    elseif isfield(modelVar, 'ClassificationSVM')
        ens = modelVar.ClassificationSVM;
    elseif isfield(modelVar, 'ClassificationKNN')
        ens = modelVar.ClassificationKNN;
    elseif isfield(modelVar, 'ClassificationTree')
        ens = modelVar.ClassificationTree;
    elseif isfield(modelVar, 'ClassificationDiscriminant')
        ens = modelVar.ClassificationDiscriminant;
    end
    if ~isempty(ens)
        try
            [~, scores] = predict(ens, feats);
            if ~isempty(scores)
                maxScore = max(scores(1, :));
                if isfinite(maxScore)
                    classConfidence = maxScore;
                end
            end
        catch
            % keep classConfidence as NaN; we'll fall back below.
        end
    end

%% 4.0 DR safety net: override Healthy/AMD -> DR when DR lesions dominate
%   The trained Bagged Trees model can miss DR. Two override paths:
%     (a) Healthy/Normal  -> DR when the COMBINED green-channel lesion
%         count exceeds OCU_DR_LESION_THRESHOLD AND at least a minimum
%         number of those are DARK (microaneurysms / dot-hemorrhages).
%     (b) AMD / Macular   -> DR when DARK lesions dominate. AMD does
%         not present with hemorrhages, so a large dark count when AMD
%         was predicted is a strong contradiction (e.g. DR_0078.jpg).
%
%   Runs BEFORE the glaucoma safety net so a real DR image with an
%   over-estimated CDR still gets labelled DR rather than Glaucoma.
    thrStr = getenv('OCU_DR_LESION_THRESHOLD');
    if isempty(thrStr)
        drThreshold = 15;
    else
        drThreshold = str2double(thrStr);
        if isnan(drThreshold) || drThreshold <= 0
            drThreshold = 15;
        end
    end
    minDarkForDR   = 8;     % at least 8 dark spots required for Healthy->DR
    drMaxCdr       = 0.60;  % DR images normally have low CDR; if CDR is
                            % above this the "DR" signal is suspicious
                            % (probably a glaucoma fundus whose dark
                            % detector picked up vessel-artefact noise).

    drOverrideFired = false;
    drOverrideReason = '';
    glaucomaOverrideFired = false;
    glaucomaOverrideReason = '';
    predLow = lower(predStr);
    fprintf('OCU_DR_DETECT: darkLesions=%d brightLesions=%d total=%d cdr=%.4f threshold=%d minDark=%d maxCdr=%.2f modelPred="%s"\n', ...
            darkLesionCount, brightLesionCount, totalLesionCount, cdr, drThreshold, minDarkForDR, drMaxCdr, predStr);

    cdrThrStr = getenv('OCU_GLAUCOMA_CDR_THRESHOLD');
    if isempty(cdrThrStr)
        glaucomaCdrThreshold = 0.65;
    else
        glaucomaCdrThreshold = str2double(cdrThrStr);
        if isnan(glaucomaCdrThreshold) || glaucomaCdrThreshold <= 0
            glaucomaCdrThreshold = 0.65;
        end
    end
    glaucomaMaxLesions = 15;  % real glaucoma fundi are quiet -- if the
                              % lesion count is high it's more likely DR
                              % with elevated CDR than true glaucoma.

    % --- (a) Healthy/Normal -> DR ---
    %   Gated on CDR < drMaxCdr so glaucoma fundi (where the dark detector
    %   often picks up vessel reflexes / choroidal patterns as "dark
    %   lesions") don't get falsely flipped to DR.
    if (contains(predLow, 'healthy') || contains(predLow, 'normal')) ...
            && totalLesionCount >= drThreshold ...
            && darkLesionCount  >= minDarkForDR ...
            && cdr              <  drMaxCdr
        fprintf('OCU_DR_OVERRIDE: forcing DR from Healthy (dark=%d>=%d, bright=%d, total=%d>=%d, cdr=%.4f<%.2f)\n', ...
                darkLesionCount, minDarkForDR, brightLesionCount, totalLesionCount, drThreshold, cdr, drMaxCdr);
        predStr = 'Diabetic Retinopathy';
        drOverrideFired = true;
        drOverrideReason = 'healthy';
    % --- (b) AMD / Macular -> DR (dark lesions dominate) ---
    elseif (contains(predLow, 'amd') || contains(predLow, 'macular')) ...
            && darkLesionCount >= max(drThreshold, 12) ...
            && darkLesionCount > brightLesionCount
        fprintf('OCU_DR_OVERRIDE: forcing DR from AMD (dark=%d > bright=%d, ge %d)\n', ...
                darkLesionCount, brightLesionCount, max(drThreshold, 12));
        predStr = 'Diabetic Retinopathy';
        drOverrideFired = true;
        drOverrideReason = 'amd';
    elseif (contains(predLow, 'healthy') || contains(predLow, 'normal')) ...
            && totalLesionCount >= drThreshold ...
            && darkLesionCount  >= minDarkForDR ...
            && cdr              >= drMaxCdr
        fprintf('OCU_DR_OVERRIDE_SKIPPED: dark=%d total=%d would have flipped to DR but cdr=%.4f >= %.2f (looks like glaucoma with noisy dark detection).\n', ...
                darkLesionCount, totalLesionCount, cdr, drMaxCdr);
    end

%% 4.1 Glaucoma safety net: override Healthy -> Glaucoma when CDR is high
%   Glaucoma's primary visual sign is optic-disc cupping (high CDR).
%   Runs after DR so a real DR image with an over-estimated CDR still
%   gets labelled DR rather than Glaucoma.
%
%   The "cdr" being tested is max(verticalCdr, sqrt(areaCdr)) capped at
%   0.85 -- see section 3e. Two gates protect against false positives:
%     1. cdr >= glaucomaCdrThreshold (default 0.65, tunable)
%     2. totalLesionCount <= glaucomaMaxLesions (real glaucoma fundi are
%        quiet; DR images polluting the cup mask with exudates inflate
%        CDR but ALSO produce many bright lesions, so a high lesion
%        count contradicts the glaucoma reading).
    predLow = lower(predStr);   % refresh after possible DR override
    fprintf('OCU_GLAUCOMA_CHECK: cdr=%.4f threshold=%.2f total=%d maxLesions=%d modelPred="%s"\n', ...
            cdr, glaucomaCdrThreshold, totalLesionCount, glaucomaMaxLesions, predStr);
    if ~drOverrideFired ...
            && (contains(predLow, 'healthy') || contains(predLow, 'normal')) ...
            && cdr >= glaucomaCdrThreshold ...
            && totalLesionCount <= glaucomaMaxLesions
        fprintf('OCU_GLAUCOMA_OVERRIDE: forcing Glaucoma (cdr=%.4f >= %.2f, total=%d <= %d)\n', ...
                cdr, glaucomaCdrThreshold, totalLesionCount, glaucomaMaxLesions);
        predStr = 'Glaucoma';
        glaucomaOverrideFired = true;
        glaucomaOverrideReason = 'cdr';
    elseif ~drOverrideFired ...
            && (contains(predLow, 'healthy') || contains(predLow, 'normal')) ...
            && cdr >= glaucomaCdrThreshold ...
            && totalLesionCount > glaucomaMaxLesions
        fprintf('OCU_GLAUCOMA_SKIPPED: cdr=%.4f >= %.2f but total=%d > %d (too many lesions for a quiet glaucoma fundus).\n', ...
                cdr, glaucomaCdrThreshold, totalLesionCount, glaucomaMaxLesions);
    end

%% 4.2 Confidence corroboration: boost score when model+heuristic agree -
%   When the trained model already voted DR and the lesion heuristic
%   independently confirms it (lesion count above threshold), treat that
%   as two corroborating signals and lift the displayed confidence so the
%   UI reflects the combined evidence rather than the raw 50% Bagged Trees
%   margin.
    modelSaysDr = contains(predLow, 'diabetic') || strcmp(predLow, 'dr') || contains(predLow, 'dr ');
    if ~drOverrideFired && modelSaysDr && totalLesionCount >= drThreshold
        saturation = min(1, max(0, (totalLesionCount - drThreshold) / (4 * drThreshold)));
        corroboratedConf = 0.85 + 0.10 * saturation;   % 85..95 %
        if isnan(classConfidence) || classConfidence < corroboratedConf
            if isnan(classConfidence)
                priorConf = 0;
            else
                priorConf = classConfidence;
            end
            fprintf('OCU_DR_CORROBORATE: model=DR + lesions=%d (>=%d) -> conf %.2f -> %.2f\n', ...
                    totalLesionCount, drThreshold, priorConf, corroboratedConf);
            classConfidence = corroboratedConf;
        end
    end

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
    % Keep `result.notes` to the concise diagnosis-specific finding
    % (matches the Python CNN engine's behaviour after the May 2026
    % readability pass). The override-reason commentary — which used to
    % be tacked onto notes as "(Bagged Trees model returned X, but N
    % candidate lesions were detected ...)" — now lives in the new
    % `result.aiReasoning` field, so the screening report shows just
    % the major image findings and the UI can offer the override
    % explanation in an expandable "AI reasoning" panel.
    aiReasoning = '';
    if drOverrideFired
        if strcmp(drOverrideReason, 'amd')
            saturation = min(1, max(0, (darkLesionCount - drThreshold) / (4 * drThreshold)));
            overrideConf = 0.65 + 0.25 * saturation;     % 65..90 %
            modelTxt = 'AMD';
        else
            saturation = min(1, max(0, (totalLesionCount - drThreshold) / (4 * drThreshold)));
            overrideConf = 0.55 + 0.30 * saturation;     % 55..85 %
            modelTxt = 'Healthy';
        end
        result.score = round(overrideConf * 1000) / 10;
        aiReasoning = sprintf( ...
            'Bagged Trees model returned %s, but %d candidate DR lesions were detected on the green channel: %d dark (microaneurysms / dot-hemorrhages) and %d bright (hard exudates / cotton-wool spots), exceeding the screening threshold of %d.', ...
            modelTxt, totalLesionCount, darkLesionCount, brightLesionCount, drThreshold);
    elseif glaucomaOverrideFired
        cdrSaturation = min(1, max(0, (cdr - glaucomaCdrThreshold) / 0.30));
        overrideConf = 0.65 + 0.27 * cdrSaturation;  % 65..92 %
        result.score = round(overrideConf * 1000) / 10;
        aiReasoning = sprintf( ...
            'Bagged Trees model returned Healthy, but the measured cup-to-disc ratio of %.4f exceeds the glaucoma screening threshold of %.2f.', ...
            cdr, glaucomaCdrThreshold);
    else
        if isnan(classConfidence)
            result.score = 90.0;
        else
            result.score = round(classConfidence * 1000) / 10;
        end
    end
    result.notes = notes;
    result.aiReasoning = aiReasoning;
    result.classConfidence = classConfidence;
    result.severity = severity;
    result.rawDiagnosis = predStr;
    result.missingFeatures = missing;
    result.drOverride = drOverrideFired;
    result.glaucomaOverride = glaucomaOverrideFired;
    result.glaucomaOverrideReason = glaucomaOverrideReason;
    result.lesionCount = totalLesionCount;
    result.darkLesionCount = darkLesionCount;
    result.brightLesionCount = brightLesionCount;

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
    features.ONH_VerticalCDR      = verticalCdr;
    features.ONH_AreaCDR          = areaCdr;
    % Disc / cup geometry normalised to the 512x512 working frame so the
    % frontend can overlay accurate disc / cup boundaries on top of the
    % uploaded fundus image regardless of its native resolution.
    features.ONH_DiscCenterX_norm = cx    / 512;
    features.ONH_DiscCenterY_norm = cy    / 512;
    features.ONH_DiscRadius_norm  = r     / 512;
    % Per-axis disc radii so the frontend can render the disc as an
    % ellipse for parity with the Python engine. The MATLAB pipeline
    % builds discMask as a perfect circle (section 3e), so X and Y
    % radii are both `r`; the keys are emitted anyway so the SVG overlay
    % uses the same code path on every engine.
    features.ONH_DiscRadiusX_norm = r     / 512;
    features.ONH_DiscRadiusY_norm = r     / 512;
    features.ONH_CupCenterX_norm  = cupCx / 512;
    features.ONH_CupCenterY_norm  = cupCy / 512;
    features.ONH_CupRadiusX_norm  = cupRx / 512;
    features.ONH_CupRadiusY_norm  = cupRy / 512;
    % Real cup contour (Nx2 array of [x,y] in [0,1]) — the frontend draws
    % this exact polygon in red, matching MATLAB's bwboundaries output. An
    % empty array makes the frontend fall back to the bbox-fit ellipse.
    features.ONH_CupBoundary      = cupBoundaryNorm;

    % --- Send the vessel display mask to the frontend as base64 PNG ----
    % The CSS SVG filter on the raw image is only an approximation; using
    % the actual binary mask we computed in 3d.b gives a much sharper view.
    % Refuse to send a degenerate (essentially empty) mask — better to let
    % the frontend fall back to its SVG filter than to show a black panel.
    features.Vessel_MaskPng = '';
    pxOn = sum(vesselsDisp(:));
    if pxOn < 250
        fprintf(2, 'OCU_WARN: vessel display mask has only %d pixels on, skipping PNG export.\n', pxOn);
    else
        tmpVesselPng = '';
        try
            tmpVesselPng = [tempname, '.png'];
            imwrite(uint8(vesselsDisp) * 255, tmpVesselPng, 'png');
            info = dir(tmpVesselPng);
            if isempty(info) || info(1).bytes <= 0
                error('OCU:VesselPng', 'imwrite produced an empty file');
            end
            fid = fopen(tmpVesselPng, 'rb');
            if fid <= 0
                error('OCU:VesselPng', 'fopen failed for vessel mask temp file');
            end
            vesselBytes = fread(fid, Inf, 'uint8=>uint8');
            fclose(fid);
            if isempty(vesselBytes)
                error('OCU:VesselPng', 'fread returned 0 bytes');
            end
            % matlab.net.base64encode exists in R2016a+ and in the
            % MATLAB Runtime, so it works in both full-MATLAB and
            % compiled-exe deployments.
            features.Vessel_MaskPng = char(matlab.net.base64encode(vesselBytes(:)));
        catch ME
            fprintf(2, 'OCU_WARN: vessel PNG encode failed: %s\n', ME.message);
            features.Vessel_MaskPng = '';
        end
        if ~isempty(tmpVesselPng) && exist(tmpVesselPng, 'file')
            delete(tmpVesselPng);
        end
    end
    features.DR_LesionCount       = totalLesionCount;
    features.DR_DarkLesionCount   = darkLesionCount;
    features.DR_BrightLesionCount = brightLesionCount;
    features.DR_LesionDensity     = totalLesionDensity;
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
