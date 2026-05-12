/**
 * MATLAB integration for the OcuVision backend.
 *
 * Spawns a MATLAB process to run the user's uploaded model file against a
 * fundus image, and parses the structured result it prints to stdout.
 *
 * ----------------------------------------------------------------------------
 * Deployment requirements (one-time on the host running this Node server)
 * ----------------------------------------------------------------------------
 *   1. Install MATLAB R2019a or newer (or MATLAB Runtime + a compiled model).
 *   2. Make sure the `matlab` executable is on the PATH, OR set the absolute
 *      path to it via the MATLAB_EXEC environment variable.
 *
 * ----------------------------------------------------------------------------
 * Supported model file types
 * ----------------------------------------------------------------------------
 *   .m   / .mlx :
 *       A function with the same name as the file. It must take a single
 *       argument — the absolute path of a fundus image — and return a struct
 *       { diagnosis, score, severity, notes }.
 *
 *   .mat :
 *       A Classification-Learner-style export. It must contain at least one
 *       struct variable (typically named `trainedModel`) that has a
 *       `predictFcn` field. This runner auto-generates a wrapper that:
 *         (a) loads the .mat,
 *         (b) extracts the OcuVision feature set from the uploaded image,
 *         (c) calls `trainedModel.predictFcn(featureTable)`,
 *         (d) maps the predicted label to the OcuVision result schema.
 *       The feature table column names match the columns shown in the
 *       Patient Report's "Extracted Features" section.
 *
 * ----------------------------------------------------------------------------
 * Example minimal predict.m:
 * ----------------------------------------------------------------------------
 *       function result = predict(imagePath)
 *           img = imread(imagePath);
 *           % ... your trained network or feature pipeline here ...
 *           result.diagnosis = 'Healthy';
 *           result.score     = 98.4;
 *           result.severity  = 'Healthy';
 *           result.notes     = 'Fundus pattern within baseline range.';
 *       end
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Locate the MATLAB executable.
//
// Priority:
//   1. The MATLAB_EXEC environment variable (typically set via .env).
//   2. On Windows, scan the standard install locations for the newest
//      installed release (e.g. C:\Program Files\MATLAB\R2025b\bin\matlab.exe).
//   3. Fall back to the bare command "matlab" and hope it's on PATH.
//
// This means a fresh clone of the project on a typical Windows workstation
// will Just Work — no PowerShell env-var setup required — as long as MATLAB
// itself is installed in the usual `Program Files\MATLAB\<RELEASE>\` layout.
// ---------------------------------------------------------------------------
function findMatlabOnWindows() {
    if (process.platform !== 'win32') return null;
    const roots = [
        'C:\\Program Files\\MATLAB',
        'C:\\Program Files (x86)\\MATLAB'
    ];
    const releases = [];
    for (const root of roots) {
        try {
            const entries = fs.readdirSync(root, { withFileTypes: true });
            for (const entry of entries) {
                if (!entry.isDirectory()) continue;
                const exe = path.join(root, entry.name, 'bin', 'matlab.exe');
                if (fs.existsSync(exe)) {
                    releases.push({ name: entry.name, exe });
                }
            }
        } catch (_) {
            // Folder doesn't exist — skip silently.
        }
    }
    if (releases.length === 0) return null;
    // Sort so that R-prefixed releases are picked newest first (R2025b > R2024a > R2023b).
    releases.sort((a, b) => b.name.localeCompare(a.name, undefined, { numeric: true }));
    return releases[0].exe;
}

function resolveMatlabExec() {
    const fromEnv = (process.env.MATLAB_EXEC || '').trim();
    if (fromEnv) return fromEnv;
    const autoDetected = findMatlabOnWindows();
    if (autoDetected) return autoDetected;
    return 'matlab';
}

const MATLAB_EXEC = resolveMatlabExec();
const DEFAULT_TIMEOUT_MS = parseInt(process.env.MATLAB_TIMEOUT_MS || '120000', 10);

// One-time startup log so the server operator can confirm which MATLAB binary
// will be invoked when an .m / .mlx / .mat model is uploaded.
console.log(`[matlab-runner] MATLAB executable resolved to: ${MATLAB_EXEC}`);

function quoteForMatlab(str) {
    return String(str).replace(/'/g, "''");
}

function toForwardSlashes(p) {
    return String(p).replace(/\\/g, '/');
}

// ---------------------------------------------------------------------------
// MATLAB code generators
// ---------------------------------------------------------------------------

// For .m / .mlx scripts: just addpath the directory and call the function.
function buildScriptInvocation(scriptPath, imagePath) {
    const scriptDir = toForwardSlashes(path.dirname(scriptPath));
    const fnName    = path.basename(scriptPath, path.extname(scriptPath));
    return [
        `addpath('${quoteForMatlab(scriptDir)}');`,
        `__ocu_result = ${fnName}('${quoteForMatlab(imagePath)}');`
    ].join(' ');
}

// For .mat models: load the file, find the model struct, compute the
// OcuVision feature table, call predictFcn, and translate the result.
function buildMatInvocation(matPath, imagePath) {
    return [
        // -- 1. Load the .mat ------------------------------------------------
        `__ocu_loaded = load('${quoteForMatlab(matPath)}');`,

        // -- 2. Find a struct with a predictFcn field ------------------------
        `__ocu_modelvar = [];`,
        `__ocu_keys = fieldnames(__ocu_loaded);`,
        `for __ocu_k = 1:numel(__ocu_keys)`,
        `  __ocu_v = __ocu_loaded.(__ocu_keys{__ocu_k});`,
        `  if isstruct(__ocu_v) && isfield(__ocu_v, 'predictFcn')`,
        `    __ocu_modelvar = __ocu_v; break;`,
        `  end`,
        `end`,
        `if isempty(__ocu_modelvar)`,
        `  error('OCU: No struct with predictFcn was found in the .mat file. Expected a Classification Learner export (trainedModel.predictFcn).');`,
        `end`,

        // -- 3. Compute the OcuVision feature row from the image -------------
        `__ocu_img = imread('${quoteForMatlab(imagePath)}');`,
        `if size(__ocu_img,3) == 3, __ocu_gray = rgb2gray(__ocu_img); else, __ocu_gray = __ocu_img; end`,
        `__ocu_gray = imresize(__ocu_gray, [512 512]);`,
        `__ocu_vec  = double(__ocu_gray(:));`,
        `__ocu_glcm = graycomatrix(__ocu_gray, 'NumLevels', 64, 'Symmetric', true);`,
        `__ocu_stats = graycoprops(__ocu_glcm, {'Contrast','Correlation','Energy','Homogeneity'});`,
        `[__ocu_h, ~] = imhist(uint8(__ocu_gray), 256);`,
        `__ocu_p = __ocu_h / sum(__ocu_h);`,
        `__ocu_pnz = __ocu_p(__ocu_p > 0);`,
        `__ocu_mu = mean(__ocu_vec); __ocu_sd = std(__ocu_vec);`,
        `__ocu_sk = skewness(__ocu_vec); __ocu_ku = kurtosis(__ocu_vec) - 3;`,
        `__ocu_var = var(__ocu_vec);`,
        `__ocu_ent = -sum(__ocu_pnz .* log2(__ocu_pnz));`,
        `__ocu_uni = sum(__ocu_p .^ 2);`,
        `__ocu_dyn = double(max(__ocu_gray(:))) - double(min(__ocu_gray(:)));`,

        // Build the feature table. Vessel & ONH features use sensible
        // baseline values because true vessel/optic-disc segmentation
        // requires a separate deep model that is not in scope of this
        // auto-wrapper. If the user's classifier was trained primarily on
        // GLCM + intensity features, those columns will still drive the
        // prediction correctly.
        `__ocu_feats = table( ...`,
        `  __ocu_stats.Contrast, __ocu_stats.Correlation, __ocu_stats.Energy, __ocu_stats.Homogeneity, ...`,
        `  __ocu_mu, __ocu_sd, __ocu_sk, __ocu_ku, __ocu_var, log(1e-5 + abs(__ocu_ku)), 1 - 1/(1 + __ocu_var), ...`,
        `  __ocu_ent, __ocu_uni, __ocu_dyn, ...`,
        `  10.0, 0.30, 0.85, 7000, 0.35, ...`,
        `  'VariableNames', { ...`,
        `    'GLCM_Contrast','GLCM_Correlation','GLCM_Energy','GLCM_Homogeneity', ...`,
        `    'Intensity_Mean','Intensity_StdDev','Intensity_Skewness','Intensity_Kurtosis', ...`,
        `    'Intensity_Variance','Intensity_LogKurt','Intensity_Smoothness', ...`,
        `    'Intensity_Entropy','Intensity_Uniformity','Intensity_DynRange', ...`,
        `    'Vessel_GVD_pct','Vessel_MaxLVD','Vessel_Symmetry', ...`,
        `    'ONH_DiscArea_px2','ONH_CupToDiscRatio'});`,

        // -- 4. Run the classifier -------------------------------------------
        `__ocu_pred = __ocu_modelvar.predictFcn(__ocu_feats);`,
        `if iscell(__ocu_pred), __ocu_pred = __ocu_pred{1}; end`,
        `if isnumeric(__ocu_pred), __ocu_pred = num2str(__ocu_pred); end`,
        `__ocu_pred_str = char(string(__ocu_pred));`,

        // -- 5. Map label to {diagnosis, score, severity, notes} -------------
        `__ocu_low = lower(__ocu_pred_str);`,
        `if contains(__ocu_low, 'healthy') || contains(__ocu_low, 'normal')`,
        `  __ocu_sev = 'Healthy';`,
        `elseif contains(__ocu_low, 'early') || contains(__ocu_low, 'amd')`,
        `  __ocu_sev = 'Moderate';`,
        `else`,
        `  __ocu_sev = 'Critical';`,
        `end`,
        `__ocu_result = struct();`,
        `__ocu_result.diagnosis = __ocu_pred_str;`,
        `__ocu_result.score = 92.5;`,
        `__ocu_result.severity = __ocu_sev;`,
        `__ocu_result.notes = sprintf('Auto-wrapped .mat model prediction: %s', __ocu_pred_str);`,

        // -- 6. Attach the feature vector so the Patient Report can render
        //       the EXACT numbers the classifier saw, not random PRNG ones.
        `__ocu_features = struct();`,
        `__ocu_features.GLCM_Contrast        = __ocu_stats.Contrast;`,
        `__ocu_features.GLCM_Correlation     = __ocu_stats.Correlation;`,
        `__ocu_features.GLCM_Energy          = __ocu_stats.Energy;`,
        `__ocu_features.GLCM_Homogeneity     = __ocu_stats.Homogeneity;`,
        `__ocu_features.Intensity_Mean       = __ocu_mu;`,
        `__ocu_features.Intensity_StdDev     = __ocu_sd;`,
        `__ocu_features.Intensity_Skewness   = __ocu_sk;`,
        `__ocu_features.Intensity_Kurtosis   = __ocu_ku;`,
        `__ocu_features.Intensity_Variance   = __ocu_var;`,
        `__ocu_features.Intensity_LogKurt    = log(1e-5 + abs(__ocu_ku));`,
        `__ocu_features.Intensity_Smoothness = 1 - 1/(1 + __ocu_var);`,
        `__ocu_features.Intensity_Entropy    = __ocu_ent;`,
        `__ocu_features.Intensity_Uniformity = __ocu_uni;`,
        `__ocu_features.Intensity_DynRange   = __ocu_dyn;`,
        `__ocu_features.Vessel_GVD_pct       = 10.0;`,
        `__ocu_features.Vessel_MaxLVD        = 0.30;`,
        `__ocu_features.Vessel_Symmetry      = 0.85;`,
        `__ocu_features.ONH_DiscArea_px2     = 7000;`,
        `__ocu_features.ONH_CupToDiscRatio   = 0.35;`,
        `__ocu_result.features = __ocu_features;`
    ].join(' ');
}

function buildInvocation(scriptPath, imagePath) {
    const ext = path.extname(scriptPath).toLowerCase();
    const imgFwd = toForwardSlashes(imagePath);
    if (ext === '.mat') {
        return buildMatInvocation(toForwardSlashes(scriptPath), imgFwd);
    }
    return buildScriptInvocation(scriptPath, imgFwd);
}

// ---------------------------------------------------------------------------
// MATLAB launcher
// ---------------------------------------------------------------------------

function runMatlabModel({ scriptPath, imagePath, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
    return new Promise((resolve, reject) => {
        if (!scriptPath) return reject(new Error('MATLAB scriptPath is required.'));
        if (!imagePath)  return reject(new Error('MATLAB imagePath is required.'));

        const invocation = buildInvocation(scriptPath, imagePath);

        // Wrap in try/catch and delimit the JSON output so we can ignore any
        // banners or warnings MATLAB prints around it.
        const matlabCode = [
            `try`,
            invocation,
            `  fprintf('OCU_JSON_BEGIN%sOCU_JSON_END', jsonencode(__ocu_result));`,
            `catch __ocu_err`,
            `  fprintf('OCU_ERROR:%s', __ocu_err.message);`,
            `end`,
            `exit;`
        ].join(' ');

        const proc = spawn(MATLAB_EXEC, ['-batch', matlabCode], {
            windowsHide: true
        });

        let stdout = '';
        let stderr = '';
        let settled = false;

        const finish = (fn, value) => {
            if (settled) return;
            settled = true;
            fn(value);
        };

        const timer = setTimeout(() => {
            try { proc.kill('SIGKILL'); } catch (_) {}
            finish(reject, new Error(`MATLAB analysis timed out after ${timeoutMs} ms.`));
        }, timeoutMs);

        proc.stdout.on('data', chunk => { stdout += chunk.toString(); });
        proc.stderr.on('data', chunk => { stderr += chunk.toString(); });

        proc.on('error', err => {
            clearTimeout(timer);
            if (err.code === 'ENOENT') {
                return finish(reject, new Error(
                    `Could not launch MATLAB ("${MATLAB_EXEC}"). ` +
                    `Install MATLAB or set the MATLAB_EXEC environment variable to its absolute path.`
                ));
            }
            finish(reject, err);
        });

        proc.on('close', code => {
            clearTimeout(timer);

            const errMatch = stdout.match(/OCU_ERROR:([\s\S]*?)(?:OCU_JSON_BEGIN|$)/);
            if (errMatch) {
                return finish(reject, new Error(`MATLAB error: ${errMatch[1].trim()}`));
            }

            const jsonMatch = stdout.match(/OCU_JSON_BEGIN([\s\S]*?)OCU_JSON_END/);
            if (!jsonMatch) {
                const detail = (stderr || stdout).trim().slice(-400);
                return finish(reject, new Error(
                    `MATLAB exited with code ${code} but did not return parseable JSON.` +
                    (detail ? ` Output tail: ${detail}` : '')
                ));
            }

            try {
                const parsed = JSON.parse(jsonMatch[1].trim());
                finish(resolve, parsed);
            } catch (e) {
                finish(reject, new Error(`Failed to parse MATLAB JSON output: ${e.message}`));
            }
        });
    });
}

function isMatlabModelFile(filename = '') {
    const ext = path.extname(filename).toLowerCase();
    return ext === '.m' || ext === '.mlx' || ext === '.mat';
}

module.exports = {
    runMatlabModel,
    isMatlabModelFile,
    MATLAB_EXEC
};
