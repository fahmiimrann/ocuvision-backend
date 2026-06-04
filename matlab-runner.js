/**
 * MATLAB integration for the OcuVision backend.
 *
 * This module shells out to either:
 *
 *   (a) a full MATLAB installation (R2019a+), driving it with `-batch` so it
 *       runs a small wrapper script (ocu_run.m) that calls the canonical
 *       OcuVision feature pipeline in ocu_predict.m, OR
 *
 *   (b) the standalone executable built by MATLAB Compiler (ocu_main.exe on
 *       Windows / ocu_main on Linux), which needs ONLY the free MATLAB
 *       Runtime to be installed on the host — no MATLAB licence required.
 *
 * The choice between the two is made automatically:
 *
 *   * If the environment variable OCU_COMPILED_EXE is set and points to an
 *     existing file, we use that.
 *   * Otherwise we resolve a full MATLAB executable (MATLAB_EXEC env var, then
 *     auto-detect under C:\Program Files\MATLAB\R*\bin\matlab.exe, then fall
 *     back to "matlab" on PATH).
 *
 * Path (a) is the iteration loop on a developer machine; path (b) is what runs
 * on the deployed server (Windows VM, GitHub Actions runner, etc.) so the
 * production server never has to hold a MATLAB licence.
 *
 * ----------------------------------------------------------------------------
 * Supported model file types
 * ----------------------------------------------------------------------------
 *   .mat                    Classification Learner export (must contain a
 *                           struct with a predictFcn field). This is the
 *                           normal case — driven by ocu_predict.m.
 *
 *   .m / .mlx               Any custom MATLAB function that takes an image
 *                           path and returns a struct with the OcuVision
 *                           result schema. Only works in "full MATLAB" mode.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const PROJECT_DIR = __dirname;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function quoteForMatlab(str) {
    return String(str).replace(/'/g, "''");
}

function toForwardSlashes(p) {
    return String(p).replace(/\\/g, '/');
}

function trimQuotes(value) {
    if (!value) return '';
    let s = String(value).trim();
    if ((s.startsWith('"') && s.endsWith('"')) ||
        (s.startsWith("'") && s.endsWith("'"))) {
        s = s.slice(1, -1).trim();
    }
    return s;
}

// ---------------------------------------------------------------------------
// Resolve the runtime: compiled .exe wins, otherwise full MATLAB.
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
        } catch (_) { /* folder doesn't exist */ }
    }
    if (releases.length === 0) return null;
    releases.sort((a, b) => b.name.localeCompare(a.name, undefined, { numeric: true }));
    return releases[0].exe;
}

function resolveMatlabExec() {
    const fromEnv = trimQuotes(process.env.MATLAB_EXEC);
    if (fromEnv) return fromEnv;
    const autoDetected = findMatlabOnWindows();
    if (autoDetected) return autoDetected;
    return 'matlab';
}

function resolveCompiledExe() {
    const fromEnv = trimQuotes(process.env.OCU_COMPILED_EXE);
    if (fromEnv && fs.existsSync(fromEnv)) return fromEnv;

    // Also accept the canonical location produced by compile_ocu.m.
    const canonical = path.join(
        PROJECT_DIR,
        'dist',
        process.platform === 'win32' ? 'ocu_main.exe' : 'ocu_main'
    );
    if (fs.existsSync(canonical)) return canonical;
    return null;
}

const COMPILED_EXE     = resolveCompiledExe();
const MATLAB_EXEC      = resolveMatlabExec();
const DEFAULT_TIMEOUT_MS = parseInt(process.env.MATLAB_TIMEOUT_MS || '120000', 10);

if (COMPILED_EXE) {
    console.log(`[matlab-runner] using compiled executable: ${COMPILED_EXE}`);
    console.log(`[matlab-runner] (no MATLAB licence required at runtime — only MATLAB Runtime)`);
} else {
    try {
        const fileVisible = MATLAB_EXEC !== 'matlab' && fs.existsSync(MATLAB_EXEC);
        console.log(
            `[matlab-runner] using full MATLAB: ${MATLAB_EXEC}` +
            ` (visible: ${fileVisible ? 'yes' : 'no'})`
        );
    } catch (_) {
        console.log(`[matlab-runner] using full MATLAB: ${MATLAB_EXEC}`);
    }
}

// ---------------------------------------------------------------------------
// Common output handling. Both code paths funnel through this — they pipe
// MATLAB stdout/stderr to the Node console (with the OCU_JSON block filtered
// out so we don't dump a wall of JSON) and resolve once the JSON arrives.
// ---------------------------------------------------------------------------

function streamAndParse({ proc, timer, settle, cleanup, label, retainOnFailureHint }) {
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', chunk => {
        const text = chunk.toString();
        stdout += text;
        const visible = text.replace(/OCU_JSON_BEGIN[\s\S]*?OCU_JSON_END/g, '<<json>>');
        process.stdout.write(`[${label}] ${visible.replace(/\n/g, `\n[${label}] `)}`);
    });
    proc.stderr.on('data', chunk => {
        const text = chunk.toString();
        stderr += text;
        process.stderr.write(`[${label}:err] ${text.replace(/\n/g, `\n[${label}:err] `)}`);
    });

    proc.on('close', code => {
        clearTimeout(timer);
        if (typeof cleanup === 'function') cleanup({ success: true });

        const errMatch = stdout.match(/OCU_ERROR:([\s\S]*?)(?:OCU_JSON_BEGIN|$)/);
        if (errMatch) {
            return settle(false, new Error(`MATLAB error: ${errMatch[1].trim()}`));
        }

        const jsonMatch = stdout.match(/OCU_JSON_BEGIN([\s\S]*?)OCU_JSON_END/);
        if (!jsonMatch) {
            const detail = (stderr || stdout).trim().slice(-400);
            return settle(false, new Error(
                `MATLAB exited with code ${code} but did not return parseable JSON.` +
                (retainOnFailureHint ? ` ${retainOnFailureHint}` : '') +
                (detail ? ` Output tail: ${detail}` : '')
            ));
        }
        try {
            settle(true, JSON.parse(jsonMatch[1].trim()));
        } catch (e) {
            settle(false, new Error(`Failed to parse MATLAB JSON output: ${e.message}`));
        }
    });
}

// ---------------------------------------------------------------------------
// Path (a): drive full MATLAB with a tiny wrapper script that calls ocu_main.
// ---------------------------------------------------------------------------

function runFullMatlab({ scriptPath, imagePath, timeoutMs }) {
    return new Promise((resolve, reject) => {
        const ext = path.extname(scriptPath).toLowerCase();

        // The wrapper script: cd into the OcuVision project dir (where
        // ocu_predict.m + ocu_main.m live), then either call ocu_main on the
        // .mat model, or addpath + call the user's custom function for
        // legacy .m / .mlx uploads.
        let invocation;
        if (ext === '.mat') {
            invocation = [
                `addpath('${quoteForMatlab(toForwardSlashes(PROJECT_DIR))}');`,
                `ocu_main('${quoteForMatlab(toForwardSlashes(scriptPath))}', '${quoteForMatlab(toForwardSlashes(imagePath))}');`
            ].join('\n');
        } else {
            // .m / .mlx — assume the function name matches the file basename
            // and that it returns the OcuVision result struct. We still
            // wrap it in the same OCU_JSON delimiters that ocu_main uses.
            const scriptDir = toForwardSlashes(path.dirname(scriptPath));
            const fnName    = path.basename(scriptPath, ext);
            invocation = [
                `addpath('${quoteForMatlab(scriptDir)}');`,
                `try`,
                `  ocu_result = ${fnName}('${quoteForMatlab(toForwardSlashes(imagePath))}');`,
                `  fprintf('OCU_JSON_BEGIN%sOCU_JSON_END', jsonencode(ocu_result));`,
                `catch ocu_err`,
                `  fprintf('OCU_ERROR:%s', ocu_err.message);`,
                `end`
            ].join('\n');
        }

        // Sanitise any stray non-ASCII characters (em-dashes, curly quotes,
        // non-breaking spaces) that could sneak in through a copy-pasted
        // folder path and confuse MATLAB's parser.
        const sanitised = invocation
            .replace(/[\u2010-\u2015]/g, '-')
            .replace(/[\u2018\u2019]/g, "'")
            .replace(/[\u201C\u201D]/g, '"')
            .replace(/\u00A0/g, ' ');

        // Write the wrapper to a temp .m file with a UTF-8 BOM so MATLAB
        // reads it deterministically regardless of system locale.
        const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ocuvision-matlab-'));
        const scriptName = 'ocu_run';
        const tempScriptPath = path.join(tempDir, `${scriptName}.m`);
        const utf8Bom = Buffer.from([0xEF, 0xBB, 0xBF]);
        fs.writeFileSync(tempScriptPath, Buffer.concat([utf8Bom, Buffer.from(sanitised, 'utf8')]));

        const cleanup = ({ success }) => {
            try {
                if (success) {
                    fs.unlinkSync(tempScriptPath);
                    fs.rmdirSync(tempDir);
                } else {
                    console.log(`[matlab-runner] temp script retained for inspection: ${tempScriptPath}`);
                }
            } catch (_) {}
        };

        const batchPayload = `cd('${quoteForMatlab(toForwardSlashes(tempDir))}'); ${scriptName}`;
        console.log(`[matlab-runner] launching MATLAB: ${tempScriptPath}`);

        const proc = spawn(MATLAB_EXEC, ['-batch', batchPayload], { windowsHide: true });

        let settled = false;
        const settle = (ok, payload) => {
            if (settled) return;
            settled = true;
            (ok ? resolve : reject)(payload);
        };

        const timer = setTimeout(() => {
            try { proc.kill('SIGKILL'); } catch (_) {}
            cleanup({ success: false });
            settle(false, new Error(`MATLAB analysis timed out after ${timeoutMs} ms.`));
        }, timeoutMs);

        proc.on('error', err => {
            clearTimeout(timer);
            cleanup({ success: false });
            if (err.code === 'ENOENT') {
                const exists = (() => { try { return fs.existsSync(MATLAB_EXEC); } catch (_) { return false; } })();
                return settle(false, new Error(
                    `Could not launch MATLAB ("${MATLAB_EXEC}"). ` +
                    (exists
                        ? `The file exists on disk but Windows refused to execute it. ` +
                          `Try opening MATLAB once from the Start menu to refresh the licence, ` +
                          `or check that anti-virus is not blocking matlab.exe.`
                        : `Node cannot see that file from its working directory. ` +
                          `Fix the MATLAB_EXEC value in .env (no surrounding quotes — write ` +
                          `MATLAB_EXEC=C:\\Program Files\\MATLAB\\R2025b\\bin\\matlab.exe).`)
                ));
            }
            settle(false, err);
        });

        streamAndParse({
            proc, timer, settle,
            cleanup: ({ success }) => cleanup({ success }),
            label: 'matlab',
            retainOnFailureHint: `Inspect: ${tempScriptPath}`
        });
    });
}

// ---------------------------------------------------------------------------
// Path (b): run the compiled standalone executable.
//
// The .exe takes two positional arguments — modelPath and imagePath — and
// prints OCU_JSON_BEGIN<json>OCU_JSON_END to stdout. It does NOT need MATLAB
// to be installed; only the MATLAB Runtime (a free, redistributable runtime
// from MathWorks). The Runtime version must match the MATLAB release used to
// build the .exe (compile_ocu.m uses your current MATLAB).
// ---------------------------------------------------------------------------

function runCompiledExe({ scriptPath, imagePath, timeoutMs }) {
    return new Promise((resolve, reject) => {
        const ext = path.extname(scriptPath).toLowerCase();
        if (ext !== '.mat') {
            // .m / .mlx scripts can't be driven through the compiled .exe.
            // Fall back to the full-MATLAB path automatically.
            return runFullMatlab({ scriptPath, imagePath, timeoutMs }).then(resolve, reject);
        }

        console.log(`[matlab-runner] launching compiled exe: ${COMPILED_EXE}`);
        const proc = spawn(COMPILED_EXE, [scriptPath, imagePath], { windowsHide: true });

        let settled = false;
        const settle = (ok, payload) => {
            if (settled) return;
            settled = true;
            (ok ? resolve : reject)(payload);
        };

        const timer = setTimeout(() => {
            try { proc.kill('SIGKILL'); } catch (_) {}
            settle(false, new Error(`Compiled OcuVision pipeline timed out after ${timeoutMs} ms.`));
        }, timeoutMs);

        proc.on('error', err => {
            clearTimeout(timer);
            if (err.code === 'ENOENT') {
                return settle(false, new Error(
                    `Could not launch the OcuVision executable ("${COMPILED_EXE}"). ` +
                    `Make sure MATLAB Runtime is installed and the .exe path in .env is correct.`
                ));
            }
            settle(false, err);
        });

        streamAndParse({
            proc, timer, settle,
            cleanup: null,
            label: 'ocu_exe',
            retainOnFailureHint: 'Did the MATLAB Runtime install correctly on this host?'
        });
    });
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

function runMatlabModel({ scriptPath, imagePath, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
    if (!scriptPath) return Promise.reject(new Error('MATLAB scriptPath is required.'));
    if (!imagePath)  return Promise.reject(new Error('MATLAB imagePath is required.'));

    if (COMPILED_EXE) {
        return runCompiledExe({ scriptPath, imagePath, timeoutMs });
    }
    return runFullMatlab({ scriptPath, imagePath, timeoutMs });
}

function isMatlabModelFile(filename = '') {
    const ext = path.extname(filename).toLowerCase();
    return ext === '.m' || ext === '.mlx' || ext === '.mat';
}

// Returns true when this host can actually run a MATLAB model — either because
// the compiled standalone .exe exists, or because MATLAB_EXEC resolved to a
// real file on disk. Used by server.js to gracefully fall back to the demo
// backend instead of hard-failing when MATLAB isn't installed on this machine.
function isMatlabAvailable() {

    if (COMPILED_EXE) return true;
    if (!MATLAB_EXEC || MATLAB_EXEC === 'matlab') return false;
    try { return fs.existsSync(MATLAB_EXEC); } catch (_) { return false; }
}

module.exports = {
    runMatlabModel,
    isMatlabModelFile,
    isMatlabAvailable,
    MATLAB_EXEC,
    COMPILED_EXE
};
