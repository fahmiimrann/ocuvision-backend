/**
 * Python inference runner for the OcuVision backend — mirror of matlab-runner.js.
 *
 * The Python engines (Option B / "python-tabular" and Option C / "python-cnn")
 * share the same wire protocol as the MATLAB runner: spawn a child process,
 * forward its stdout/stderr to the Node console with the JSON block filtered
 * out, then resolve once the marker delimiters appear:
 *
 *     PY_JSON_BEGIN{ ...json... }PY_JSON_END
 *
 * Errors are surfaced as:
 *
 *     PY_ERROR:<message>
 *
 * The interpreter is resolved in this order (first hit wins):
 *
 *   1. OCU_PYTHON_EXEC env var (absolute path to python.exe)
 *   2. ./python/.venv/Scripts/python.exe  (Windows venv next to the project)
 *   3. ./python/.venv/bin/python          (POSIX venv next to the project)
 *   4. "python" on PATH                   (last-ditch fallback)
 *
 * Models live next to the project under ./models/. Override with
 * OCU_MODELS_DIR if you keep them somewhere else (e.g. inside the
 * unet_project sibling folder).
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const PROJECT_DIR = __dirname;
const PY_DIR      = path.join(PROJECT_DIR, 'python');
const DEFAULT_MODELS_DIR = path.join(PROJECT_DIR, 'models');

function trimQuotes(value) {
    if (!value) return '';
    let s = String(value).trim();
    if ((s.startsWith('"') && s.endsWith('"')) ||
        (s.startsWith("'") && s.endsWith("'"))) {
        s = s.slice(1, -1).trim();
    }
    return s;
}

function resolvePythonExec() {
    const fromEnv = trimQuotes(process.env.OCU_PYTHON_EXEC);
    if (fromEnv) return fromEnv;

    const candidates = [
        path.join(PY_DIR, '.venv', 'Scripts', 'python.exe'), // Windows venv
        path.join(PY_DIR, '.venv', 'bin', 'python'),         // POSIX venv
    ];
    for (const c of candidates) {
        try { if (fs.existsSync(c)) return c; } catch (_) { /* keep trying */ }
    }
    return 'python';
}

function resolveModelsDir() {
    const fromEnv = trimQuotes(process.env.OCU_MODELS_DIR);
    if (fromEnv) return fromEnv;
    return DEFAULT_MODELS_DIR;
}

const PYTHON_EXEC = resolvePythonExec();
const MODELS_DIR  = resolveModelsDir();
const DEFAULT_TIMEOUT_MS = parseInt(process.env.PYTHON_TIMEOUT_MS || '180000', 10);

(() => {
    try {
        const exists = PYTHON_EXEC !== 'python' && fs.existsSync(PYTHON_EXEC);
        console.log(`[python-runner] interpreter: ${PYTHON_EXEC}` +
                    ` (visible: ${exists ? 'yes' : 'no'})`);
    } catch (_) {
        console.log(`[python-runner] interpreter: ${PYTHON_EXEC}`);
    }
    console.log(`[python-runner] models dir : ${MODELS_DIR}`);
})();

// ---------------------------------------------------------------------------
// stdout/stderr handler shared by both engines
// ---------------------------------------------------------------------------
function streamAndParse({ proc, timer, settle, label }) {
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', chunk => {
        const text = chunk.toString();
        stdout += text;
        const visible = text.replace(/PY_JSON_BEGIN[\s\S]*?PY_JSON_END/g, '<<json>>');
        process.stdout.write(`[${label}] ${visible.replace(/\n/g, `\n[${label}] `)}`);
    });
    proc.stderr.on('data', chunk => {
        const text = chunk.toString();
        stderr += text;
        process.stderr.write(`[${label}:err] ${text.replace(/\n/g, `\n[${label}:err] `)}`);
    });

    proc.on('close', code => {
        clearTimeout(timer);

        const errMatch = stdout.match(/PY_ERROR:([\s\S]*?)(?:PY_JSON_BEGIN|$)/);
        if (errMatch) {
            return settle(false, new Error(`Python error: ${errMatch[1].trim()}`));
        }

        const jsonMatch = stdout.match(/PY_JSON_BEGIN([\s\S]*?)PY_JSON_END/);
        if (!jsonMatch) {
            const detail = (stderr || stdout).trim().slice(-400);
            return settle(false, new Error(
                `Python exited with code ${code} but did not return parseable JSON.` +
                (detail ? ` Output tail: ${detail}` : '')
            ));
        }
        try {
            settle(true, JSON.parse(jsonMatch[1].trim()));
        } catch (e) {
            settle(false, new Error(`Failed to parse Python JSON output: ${e.message}`));
        }
    });
}

// ---------------------------------------------------------------------------
// Public entry point
//
//   runPythonModel({ engine: 'tabular' | 'cnn', imagePath, timeoutMs? })
//     -> Promise<resultJson>
// ---------------------------------------------------------------------------
function runPythonModel({ engine, imagePath, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
    return new Promise((resolve, reject) => {
        if (!engine)    return reject(new Error('python engine is required.'));
        if (!imagePath) return reject(new Error('python imagePath is required.'));
        if (!['tabular', 'cnn'].includes(engine)) {
            return reject(new Error(`Unsupported python engine: ${engine}`));
        }

        const script = path.join(PY_DIR, 'python_predict.py');
        if (!fs.existsSync(script)) {
            return reject(new Error(`python_predict.py not found at ${script}`));
        }

        const args = [
            script,
            '--engine', engine,
            '--image',  imagePath,
            '--models-dir', MODELS_DIR,
        ];

        console.log(`[python-runner] launching: ${PYTHON_EXEC} ${args.join(' ')}`);

        const proc = spawn(PYTHON_EXEC, args, {
            windowsHide: true,
            cwd: PY_DIR,                  // so 'import unet' / 'import feature_extraction' resolve
            env: { ...process.env, PYTHONUNBUFFERED: '1' },
        });

        let settled = false;
        const settle = (ok, payload) => {
            if (settled) return;
            settled = true;
            (ok ? resolve : reject)(payload);
        };

        const timer = setTimeout(() => {
            try { proc.kill('SIGKILL'); } catch (_) {}
            settle(false, new Error(`Python ${engine} pipeline timed out after ${timeoutMs} ms.`));
        }, timeoutMs);

        proc.on('error', err => {
            clearTimeout(timer);
            if (err.code === 'ENOENT') {
                return settle(false, new Error(
                    `Could not launch Python ("${PYTHON_EXEC}"). ` +
                    `Make sure the interpreter exists and pip install -r python/requirements.txt has been run. ` +
                    `Override with OCU_PYTHON_EXEC=<absolute path> in .env.`
                ));
            }
            settle(false, err);
        });

        streamAndParse({ proc, timer, settle, label: `py-${engine}` });
    });
}

function isPythonEngineType(type = '') {
    return type === 'python-tabular' || type === 'python-cnn';
}

function engineFromType(type = '') {
    if (type === 'python-tabular') return 'tabular';
    if (type === 'python-cnn')     return 'cnn';
    return null;
}

module.exports = {
    runPythonModel,
    isPythonEngineType,
    engineFromType,
    PYTHON_EXEC,
    MODELS_DIR,
};
