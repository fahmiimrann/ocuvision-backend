/**
 * MATLAB integration for the OcuVision backend.
 *
 * This module spawns a MATLAB process to run the user's uploaded model file
 * (.m or .mlx) against a fundus image, and parses the structured result it
 * prints to stdout.
 *
 * ----------------------------------------------------------------------------
 * Deployment requirements (one-time on the host running this Node server):
 * ----------------------------------------------------------------------------
 *   1. Install MATLAB R2019a or newer (or MATLAB Runtime + a compiled model).
 *   2. Make sure the `matlab` executable is on the PATH, OR set the absolute
 *      path to it via the MATLAB_EXEC environment variable.
 *
 * ----------------------------------------------------------------------------
 * Contract that the user's MATLAB script must satisfy:
 * ----------------------------------------------------------------------------
 *   - It must be a function with the same name as the file (without extension).
 *   - It must take a single argument: the absolute path of a fundus image.
 *   - It must return a struct with these fields:
 *
 *         result.diagnosis  : char/string  e.g. 'Diabetic Retinopathy'
 *         result.score      : double 0–100 (confidence percentage)
 *         result.severity   : char/string  one of 'Healthy' | 'Moderate' | 'Critical'
 *         result.notes      : char/string  short clinical interpretation
 *
 *   Example minimal predict.m:
 *
 *       function result = predict(imagePath)
 *           img = imread(imagePath);
 *           % ... your trained network or feature pipeline here ...
 *           result.diagnosis = 'Healthy';
 *           result.score     = 98.4;
 *           result.severity  = 'Healthy';
 *           result.notes     = 'Fundus pattern within baseline range.';
 *       end
 *
 * The Node server invokes the script with `matlab -batch` and reads the JSON
 * result delimited by OCU_JSON_BEGIN ... OCU_JSON_END markers, so the user's
 * script does not need to worry about printing anything itself.
 * ----------------------------------------------------------------------------
 */

const { spawn } = require('child_process');
const path = require('path');

const MATLAB_EXEC = process.env.MATLAB_EXEC || 'matlab';
const DEFAULT_TIMEOUT_MS = parseInt(process.env.MATLAB_TIMEOUT_MS || '120000', 10);

function quoteForMatlab(str) {
    return String(str).replace(/'/g, "''");
}

function toForwardSlashes(p) {
    return String(p).replace(/\\/g, '/');
}

function runMatlabModel({ scriptPath, imagePath, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
    return new Promise((resolve, reject) => {
        if (!scriptPath) return reject(new Error('MATLAB scriptPath is required.'));
        if (!imagePath)  return reject(new Error('MATLAB imagePath is required.'));

        const scriptDir = toForwardSlashes(path.dirname(scriptPath));
        const fnName    = path.basename(scriptPath, path.extname(scriptPath));
        const imgPathFwd = toForwardSlashes(imagePath);

        // We wrap the model call in a try/catch and delimit the JSON output so
        // we can ignore any banners or warnings MATLAB prints around it.
        const matlabCode = [
            `addpath('${quoteForMatlab(scriptDir)}');`,
            `try`,
            `  __ocu_result = ${fnName}('${quoteForMatlab(imgPathFwd)}');`,
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
    return ext === '.m' || ext === '.mlx';
}

module.exports = {
    runMatlabModel,
    isMatlabModelFile,
    MATLAB_EXEC
};
