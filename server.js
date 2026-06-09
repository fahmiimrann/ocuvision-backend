const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
require('dotenv').config();

const db = require('./db');
const { runMatlabModel, isMatlabModelFile, isMatlabAvailable, MATLAB_EXEC, COMPILED_EXE } = require('./matlab-runner');
const { runPythonModel, isPythonEngineType, engineFromType, PYTHON_EXEC, MODELS_DIR: PY_MODELS_DIR } = require('./python-runner');

const app = express();
const PORT = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage() });

// --- Storage paths (model registry only — user/record data lives in db.js) ---
const MODEL_DIR = path.join(__dirname, 'uploaded-models');
const MODEL_REGISTRY_FILE = path.join(MODEL_DIR, 'models.json');

const DEFAULT_MODEL = {
    id: 'demo-retina-core',
    name: 'Demo Retina Core',
    originalName: 'Built-in demo backend logic',
    type: 'demo',
    size: 0,
    uploadedAt: '2026-05-07T00:00:00.000Z',
    active: true
};

// Built-in Python pseudo-models. These aren't uploaded by the user; they
// represent the in-tree inference pipelines under ./python/*. We ensure
// they're present in models.json on every boot so the UI can list them as
// selectable engines alongside the MATLAB Bagged Trees and any other
// uploaded .mat models.
const BUILTIN_PYTHON_MODELS = [
    {
        id: 'python-cnn-resnet50',
        name: 'Python AI - End-to-End ResNet50 CNN',
        originalName: 'Built-in (python/python_predict.py --engine cnn)',
        type: 'python-cnn',
        size: 0,
        uploadedAt: '2026-05-13T00:00:00.000Z',
        active: false,
        builtin: true
    }
];

// --- Auth helpers ----------------------------------------------------------
function generateToken() {
    return crypto.randomBytes(32).toString('hex');
}

function publicUser(user) {
    if (!user) return null;
    const { password, token, ...safe } = user;
    return safe;
}

async function requireAuth(req, res, next) {
    try {
        const header = req.headers.authorization || '';
        const token = header.startsWith('Bearer ') ? header.slice(7).trim() : null;
        const user = await db.findUserByToken(token);
        if (!user) return res.status(401).json({ error: 'Authentication required.' });
        req.user = user;
        req.token = token;
        next();
    } catch (err) {
        next(err);
    }
}

function ensureModelRegistry() {
    if (!fs.existsSync(MODEL_DIR)) {
        fs.mkdirSync(MODEL_DIR, { recursive: true });
    }

    if (!fs.existsSync(MODEL_REGISTRY_FILE)) {
        fs.writeFileSync(MODEL_REGISTRY_FILE,
            JSON.stringify([DEFAULT_MODEL, ...BUILTIN_PYTHON_MODELS], null, 2));
    }
}

// Merge any built-in models we ship with the project into the on-disk
// registry. Idempotent: if a built-in id already exists we leave it alone
// (so the user's `active` choice survives a server restart). Returns the
// possibly-updated array; caller decides whether to writeModels(...) it.
function withBuiltinsMerged(models) {
    const byId = new Map(models.map(m => [m.id, m]));
    let changed = false;
    for (const builtin of [DEFAULT_MODEL, ...BUILTIN_PYTHON_MODELS]) {
        if (!byId.has(builtin.id)) {
            models.push(builtin);
            byId.set(builtin.id, builtin);
            changed = true;
        }
    }
    return { models, changed };
}

function readModels() {
    ensureModelRegistry();
    try {
        const raw = JSON.parse(fs.readFileSync(MODEL_REGISTRY_FILE, 'utf8'));
        const arr = Array.isArray(raw) && raw.length ? raw : [DEFAULT_MODEL];
        const { models, changed } = withBuiltinsMerged(arr.slice());
        if (changed) writeModels(models);
        return models;
    } catch (err) {
        return [DEFAULT_MODEL, ...BUILTIN_PYTHON_MODELS];
    }
}

function writeModels(models) {
    ensureModelRegistry();
    fs.writeFileSync(MODEL_REGISTRY_FILE, JSON.stringify(models, null, 2));
}

function getActiveModel() {
    return readModels().find(model => model.active) || DEFAULT_MODEL;
}

const modelUpload = multer({
    storage: multer.diskStorage({
        destination: function (_req, _file, cb) {
            ensureModelRegistry();
            cb(null, MODEL_DIR);
        },
        filename: function (_req, file, cb) {
            const safeBase = path.basename(file.originalname).replace(/[^a-z0-9._-]/gi, '_');
            cb(null, `${Date.now()}-${safeBase}`);
        }
    }),
    limits: { fileSize: 100 * 1024 * 1024 }
});

/**
 * CORS Configuration
 * Allows your GitHub Pages frontend to communicate with this Render backend.
 */
const allowedOrigins = [
    'http://localhost:5500', // For local development testing
    'http://127.0.0.1:5500',
    'http://localhost:3000',
    'https://calisto.github.io',
    'https://fahmiimrann.github.io',
    'https://fahmiimrann.github.io/calisto.github.io',
    'https://fahmimrann.github.io',
    'https://fahmimrann.github.io/calisto.github.io' // Your live frontend
];

app.use(cors({
    origin: function (origin, callback) {
        // Allow requests with no origin (like mobile apps or curl)
        if (!origin) return callback(null, true);
        if (allowedOrigins.indexOf(origin) === -1) {
            const msg = 'The CORS policy for this site does not allow access from the specified Origin.';
            return callback(new Error(msg), false);
        }
        return callback(null, true);
    }
}));

// Bumped to 25 MB so a base64 fundus image (typically 1–3 MB) easily fits
// inside the record JSON body sent by the frontend.
app.use(express.json({ limit: '25mb' }));

// Serve the OcuVision frontend (index.html + Calisto Logo.png + Background_Video.mp4
// + any other root-level assets) from this Node process. This is what lets the
// user open http://localhost:3000 and run the site against the LOCAL backend,
// which is the only environment that can actually call MATLAB (GitHub Pages and
// Render cannot host MATLAB). API routes below take precedence over file
// matches with the same path because express.static only handles existing files
// and falls through to the next handler otherwise.
//
// `dotfiles: 'deny'` prevents serving `.env`, `.git`, etc. The remaining JS/json
// files in the folder (server.js, db.js, matlab-runner.js, package.json) are
// the same source already published on GitHub, so exposing them locally is
// harmless.
app.use(express.static(__dirname, {
    index: 'index.html',
    dotfiles: 'deny',
    setHeaders(res, filePath) {
        if (filePath.endsWith('index.html')) {
            res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
        }
    }
}));

// Health-check / API status (used to live at GET /, but / now serves index.html).
app.get('/api/health', (req, res) => {
    let matlabRuntime;
    if (COMPILED_EXE) {
        matlabRuntime = { mode: 'compiled', executable: COMPILED_EXE };
    } else {
        matlabRuntime = { mode: 'full-matlab', executable: MATLAB_EXEC };
    }

    const tabularReady = fs.existsSync(path.join(PY_MODELS_DIR, 'tabular_clf.pkl'));
    const cnnReady     = fs.existsSync(path.join(PY_MODELS_DIR, 'cnn_clf.pth'));

    res.json({
        status: "Online",
        project: "OcuVision AI API",
        message: "Welcome to the Ophthalmic Intelligence Backend",
        activeModel: getActiveModel().name,
        runtime: matlabRuntime,
        matlabConfigured: Boolean(MATLAB_EXEC) || Boolean(COMPILED_EXE),
        python: {
            interpreter: PYTHON_EXEC,
            modelsDir: PY_MODELS_DIR,
            tabularReady,
            cnnReady
        }
    });
});

// Login: validates against persisted users and issues a session token.
app.post('/api/login', async (req, res, next) => {
    try {
        const { username, password } = req.body || {};
        if (!username || !password) {
            return res.status(400).json({ error: 'Username and password are required.' });
        }

        const user = await db.findUserByUsername(username);
        if (!user || user.password !== password) {
            return res.status(401).json({ error: 'Invalid username or password.' });
        }

        const token = generateToken();
        const updated = await db.setUserToken(user.id, token);
        res.json({ user: publicUser(updated), token });
    } catch (err) {
        next(err);
    }
});

// Register: persists the new user and issues a session token.
app.post('/api/register', async (req, res, next) => {
    try {
        const { username, name, type, password, email } = req.body || {};
        if (!username || !name || !type || !password) {
            return res.status(400).json({ error: 'Name, username, password, and type are required.' });
        }

        const existing = await db.findUserByUsername(username);
        if (existing) return res.status(409).json({ error: 'Username is already taken.' });

        const token = generateToken();
        const user = {
            id: `u-${Date.now()}`,
            username,
            password,
            name,
            type,
            email: email || `${username}@calisto.com`,
            avatar: '',
            created_at: new Date().toISOString(),
            token
        };
        const inserted = await db.insertUser(user);
        res.status(201).json({ user: publicUser(inserted), token });
    } catch (err) {
        next(err);
    }
});

// Logout: clears the session token for the current user.
app.post('/api/logout', requireAuth, async (req, res, next) => {
    try {
        await db.setUserToken(req.user.id, null);
        res.json({ ok: true });
    } catch (err) {
        next(err);
    }
});

// Re-verify the current user's password (used before destructive actions).
app.post('/api/auth/verify', requireAuth, (req, res) => {
    const { password } = req.body || {};
    if (!password) return res.status(400).json({ error: 'Password is required.' });
    if (req.user.password !== password) {
        return res.status(401).json({ error: 'Incorrect password.' });
    }
    res.json({ ok: true });
});

// Return the currently authenticated user.
app.get('/api/users/me', requireAuth, (req, res) => {
    res.json({ user: publicUser(req.user) });
});

// Update the authenticated user's profile (name, username, email, avatar).
app.patch('/api/users/me', requireAuth, async (req, res, next) => {
    try {
        const { name, username, email, avatar } = req.body || {};
        const patch = {};

        if (username && username !== req.user.username) {
            const conflict = await db.findUserByUsername(username);
            if (conflict && conflict.id !== req.user.id) {
                return res.status(409).json({ error: 'Username is already taken.' });
            }
            patch.username = username;
        }
        if (typeof name === 'string'   && name.trim())   patch.name = name.trim();
        if (typeof email === 'string')                   patch.email = email.trim();
        if (typeof avatar === 'string')                  patch.avatar = avatar;

        const updated = await db.patchUser(req.user.id, patch);
        if (!updated) return res.status(404).json({ error: 'User not found.' });
        res.json({ user: publicUser(updated) });
    } catch (err) {
        next(err);
    }
});

// Change the authenticated user's password (requires current password).
app.post('/api/users/me/password', requireAuth, async (req, res, next) => {
    try {
        const { currentPassword, newPassword } = req.body || {};
        if (!currentPassword || !newPassword) {
            return res.status(400).json({ error: 'Current and new password are required.' });
        }
        if (req.user.password !== currentPassword) {
            return res.status(401).json({ error: 'Current password is incorrect.' });
        }
        if (newPassword.length < 4) {
            return res.status(400).json({ error: 'New password must be at least 4 characters.' });
        }

        await db.patchUser(req.user.id, { password: newPassword });
        res.json({ ok: true });
    } catch (err) {
        next(err);
    }
});

// --- Demo result helpers ---------------------------------------------------
// Baseline feature vectors per condition. These mirror the ranges used by
// the frontend report generator so that, even in demo mode, the Patient
// Report shows numbers consistent with the diagnosis. When a real MATLAB
// model is active these values are replaced by the actual computed
// feature vector returned by the model.
const DEMO_RESULTS = [
    {
        status: 'Alert',
        score: '94.8',
        result: 'Diabetic Retinopathy',
        severity: 'Critical',
        notes: 'Multiple microaneurysms and intraretinal hemorrhages detected. Expert validation recommended.',
        features: {
            GLCM_Contrast: 71.4, GLCM_Correlation: 0.886, GLCM_Energy: 0.108, GLCM_Homogeneity: 0.461,
            Intensity_Mean: 122.5, Intensity_StdDev: 66.1, Intensity_Skewness: 0.83, Intensity_Kurtosis: 0.22,
            Intensity_Variance: 4369.21, Intensity_LogKurt: -1.514, Intensity_Smoothness: 0.99977,
            Intensity_Entropy: 7.61, Intensity_Uniformity: 0.0061, Intensity_DynRange: 252,
            Vessel_GVD_pct: 9.5, Vessel_MaxLVD: 0.38, Vessel_Symmetry: 0.70,
            ONH_DiscArea_px2: 6800, ONH_CupToDiscRatio: 0.34
        }
    },
    {
        status: 'Warning',
        score: '89.3',
        result: 'Early Glaucoma indicators detected',
        severity: 'Moderate',
        notes: 'Cup-to-disc ratio and vessel morphology should be reviewed by an ophthalmologist.',
        features: {
            GLCM_Contrast: 42.6, GLCM_Correlation: 0.948, GLCM_Energy: 0.151, GLCM_Homogeneity: 0.547,
            Intensity_Mean: 118.7, Intensity_StdDev: 51.4, Intensity_Skewness: 0.28, Intensity_Kurtosis: -0.78,
            Intensity_Variance: 2641.96, Intensity_LogKurt: -0.248, Intensity_Smoothness: 0.99962,
            Intensity_Entropy: 7.31, Intensity_Uniformity: 0.0079, Intensity_DynRange: 249,
            Vessel_GVD_pct: 10.2, Vessel_MaxLVD: 0.31, Vessel_Symmetry: 0.91,
            ONH_DiscArea_px2: 7400, ONH_CupToDiscRatio: 0.52
        }
    },
    {
        status: 'Optimal',
        score: '98.7',
        result: 'Healthy',
        severity: 'Healthy',
        notes: 'Fundus pattern is within baseline range for this demo backend screening.',
        features: {
            GLCM_Contrast: 34.5, GLCM_Correlation: 0.953, GLCM_Energy: 0.156, GLCM_Homogeneity: 0.552,
            Intensity_Mean: 118.4, Intensity_StdDev: 52.0, Intensity_Skewness: 0.25, Intensity_Kurtosis: -0.81,
            Intensity_Variance: 2704.00, Intensity_LogKurt: -0.211, Intensity_Smoothness: 0.99963,
            Intensity_Entropy: 7.32, Intensity_Uniformity: 0.0078, Intensity_DynRange: 250,
            Vessel_GVD_pct: 11.6, Vessel_MaxLVD: 0.31, Vessel_Symmetry: 0.93,
            ONH_DiscArea_px2: 6800, ONH_CupToDiscRatio: 0.29
        }
    }
];

function pickDemoResult() {
    return DEMO_RESULTS[Math.floor(Math.random() * DEMO_RESULTS.length)];
}

function deriveStatus(severity = '', result = '') {
    const sev = String(severity).toLowerCase();
    const res = String(result).toLowerCase();
    if (sev.includes('critical') || res.includes('retinopathy') || res.includes('glaucoma')) return 'Alert';
    if (sev.includes('moderate') || res.includes('early') || res.includes('amd')) return 'Warning';
    return 'Optimal';
}

function writeTempImage(buffer, originalName) {
    const safeBase = path.basename(originalName || 'fundus.png').replace(/[^a-z0-9._-]/gi, '_');
    const tmpPath = path.join(os.tmpdir(), `ocu-${Date.now()}-${crypto.randomBytes(4).toString('hex')}-${safeBase}`);
    fs.writeFileSync(tmpPath, buffer);
    return tmpPath;
}

// AI analysis endpoint used by the diagnostics image upload flow.
//
// The frontend may include a `modelId` form field to choose which engine
// runs for this specific scan (per-image picker on the upload page).
// When `modelId` is omitted we fall back to the currently active model
// from Settings -> AI Core.
//
// Routing:
//   * Built-in python-tabular / python-cnn  -> runPythonModel(...)
//   * Uploaded .mat / .m / .mlx             -> runMatlabModel(...)
//   * Everything else                       -> demo result
app.post('/api/analyze', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded.' });
    }

    const requestedId = (req.body?.modelId || '').toString().trim();
    const models = readModels();
    const selectedModel = (requestedId && models.find(m => m.id === requestedId)) || getActiveModel();
    const modelName = selectedModel?.name || 'Demo Retina Core';
    const storedFile = selectedModel?.storedFile
        ? path.join(MODEL_DIR, selectedModel.storedFile)
        : null;

    let tempImagePath = null;
    try {
        // ---- Python engines (built-in: tabular + CNN) ----
        if (isPythonEngineType(selectedModel?.type)) {
            tempImagePath = writeTempImage(req.file.buffer, req.file.originalname);
            const engine = engineFromType(selectedModel.type);
            const pyRaw = await runPythonModel({ engine, imagePath: tempImagePath });

            const diagnosis = pyRaw.diagnosis || pyRaw.result || 'Healthy';
            const scoreNum  = Number(pyRaw.score ?? pyRaw.confidence ?? 0);
            const scoreStr  = Number.isFinite(scoreNum) ? scoreNum.toFixed(1) : '0.0';
            const severity  = pyRaw.severity || 'Healthy';
            const notes     = pyRaw.notes || 'Python AI model analysis completed.';
            const status    = pyRaw.status || deriveStatus(severity, diagnosis);
            const features  = pyRaw.features && typeof pyRaw.features === 'object' ? pyRaw.features : null;

            return res.json({
                status,
                score: scoreStr,
                result: diagnosis,
                severity,
                notes,
                confidence: `${scoreStr}%`,
                fileName: req.file.originalname,
                features,
                probs: pyRaw.probs || null,
                model: {
                    id: selectedModel.id,
                    name: modelName,
                    runtime: selectedModel.type,
                    kind: pyRaw.modelKind || selectedModel.type
                }
            });
        }

        // ---- MATLAB engines (.m / .mlx / .mat) ----
        if (
            storedFile &&
            fs.existsSync(storedFile) &&
            isMatlabModelFile(selectedModel.originalName || storedFile)
        ) {
            // If this host has no MATLAB / MATLAB Runtime installed, don't even
            // try to spawn it — fall back to the demo backend with a transparent
            // note so the website stays usable on machines without MATLAB.
            if (!isMatlabAvailable()) {
                console.warn(
                    `[analyze] MATLAB is not available on this host ` +
                    `(MATLAB_EXEC="${MATLAB_EXEC}", compiledExe=${Boolean(COMPILED_EXE)}). ` +
                    `Falling back to demo backend for model "${modelName}".`
                );
                const fallback = pickDemoResult();
                return res.json({
                    ...fallback,
                    notes: `${fallback.notes} (MATLAB not detected on this machine — demo backend used instead of "${modelName}". Install MATLAB or switch the active engine to Demo Retina Core to remove this notice.)`,
                    confidence: `${fallback.score}%`,
                    fileName: req.file.originalname,
                    features: fallback.features || null,
                    model: {
                        id: selectedModel.id,
                        name: modelName,
                        runtime: 'demo-fallback',
                        requestedRuntime: 'matlab'
                    }
                });
            }

            tempImagePath = writeTempImage(req.file.buffer, req.file.originalname);
            const matlabRaw = await runMatlabModel({
                scriptPath: storedFile,
                imagePath: tempImagePath
            });

            const diagnosis = matlabRaw.diagnosis || matlabRaw.result || 'Healthy';
            const scoreNum  = Number(matlabRaw.score ?? matlabRaw.confidence ?? 0);
            const scoreStr  = Number.isFinite(scoreNum) ? scoreNum.toFixed(1) : '0.0';
            const severity  = matlabRaw.severity || 'Healthy';
            const notes     = matlabRaw.notes || 'MATLAB model analysis completed.';
            const status    = matlabRaw.status || deriveStatus(severity, diagnosis);
            const features  = matlabRaw.features && typeof matlabRaw.features === 'object'
                ? matlabRaw.features
                : null;

            // Surface partial-prediction info to the server log only - it is
            // useful for developers (so they know to supply formulas for the
            // missing features) but should not appear in clinician-facing UI.
            if (Array.isArray(matlabRaw.missingFeatures) && matlabRaw.missingFeatures.length > 0) {
                console.warn(
                    `[analyze] Model "${modelName}" returned a partial prediction: ` +
                    `${matlabRaw.missingFeatures.length} feature(s) defaulted to 0 - ${matlabRaw.missingFeatures.join(', ')}. ` +
                    `Raw class: "${matlabRaw.rawDiagnosis || diagnosis}".`
                );
            }

            return res.json({
                status,
                score: scoreStr,
                result: diagnosis,
                severity,
                notes,
                confidence: `${scoreStr}%`,
                fileName: req.file.originalname,
                features,
                model: { id: selectedModel.id, name: modelName, runtime: 'matlab' }
            });
        }

        // ---- Demo / unsupported types ----
        const result = pickDemoResult();
        return res.json({
            ...result,
            confidence: `${result.score}%`,
            fileName: req.file.originalname,
            features: result.features || null,
            model: { id: selectedModel.id, name: modelName, runtime: selectedModel.type || 'demo' }
        });
    } catch (err) {
        console.error(`[analyze] ${modelName} pipeline failed:`, err.message);
        return res.status(502).json({
            error: `AI model "${modelName}" failed to run: ${err.message}`,
            model: { id: selectedModel?.id, name: modelName }
        });
    } finally {
        if (tempImagePath) {
            fs.unlink(tempImagePath, () => {});
        }
    }
});

// Model registry endpoints for Preferences > AI Core.
app.get('/api/models', (_req, res) => {
    res.json({
        models: readModels(),
        activeModel: getActiveModel()
    });
});

app.post('/api/models', modelUpload.single('model'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No model file uploaded.' });
    }

    const models = readModels().map(model => ({ ...model, active: false }));
    const model = {
        id: `model-${Date.now()}`,
        name: req.body.name?.trim() || path.parse(req.file.originalname).name,
        originalName: req.file.originalname,
        storedFile: req.file.filename,
        type: path.extname(req.file.originalname).replace('.', '').toLowerCase() || 'model',
        size: req.file.size,
        uploadedAt: new Date().toISOString(),
        active: true
    };

    models.push(model);
    writeModels(models);

    res.status(201).json({
        model,
        models,
        message: `${model.name} uploaded and activated.`
    });
});

app.post('/api/models/select', (req, res) => {
    const { id } = req.body;
    const models = readModels();

    if (!models.some(model => model.id === id)) {
        return res.status(404).json({ error: 'Model not found.' });
    }

    const updatedModels = models.map(model => ({
        ...model,
        active: model.id === id
    }));

    writeModels(updatedModels);

    res.json({
        activeModel: updatedModels.find(model => model.id === id),
        models: updatedModels
    });
});

// --- Patient records CRUD --------------------------------------------------
// Record IDs are condition-based so the registry stays human-readable:
//   Healthy                 -> OCU-H00001, OCU-H00002, ...
//   AMD / Macular Degen.    -> OCU-AMD00001, ...
//   Diabetic Retinopathy    -> OCU-DR00001, ...
//   Glaucoma (any stage)    -> OCU-G00001, ...
// The numeric suffix is a 5-digit, zero-padded, monotonically increasing
// counter scoped to that prefix. We compute it by scanning existing records,
// which keeps the logic simple and works with both Supabase + local JSON.
function getRecordIdPrefix(result, severity) {
    const r = String(result || '').toLowerCase();
    const s = String(severity || '').toLowerCase();
    if (r.includes('amd') || r.includes('macular')) return 'OCU-AMD';
    if (r.includes('diabetic') || r.includes('retinopathy') || /\bdr\b/.test(r)) return 'OCU-DR';
    if (r.includes('glaucoma')) return 'OCU-G';
    if (s === 'healthy' || r.includes('healthy') || r.includes('normal') || r.includes('no urgent')) return 'OCU-H';
    return 'OCU-H';
}

async function nextRecordId(result, severity) {
    const prefix = getRecordIdPrefix(result, severity);
    const records = await db.getAllRecords();
    const pattern = new RegExp(`^${prefix}(\\d+)$`);
    let max = 0;
    for (const r of records) {
        const m = pattern.exec(r && r.id ? String(r.id) : '');
        if (m) {
            const n = parseInt(m[1], 10);
            if (Number.isFinite(n) && n > max) max = n;
        }
    }
    return `${prefix}${String(max + 1).padStart(5, '0')}`;
}

function normalizeRecordPayload(body = {}) {
    const allowed = ['patient', 'age', 'gender', 'date', 'result', 'confidence', 'doctor', 'severity', 'fundus_image', 'features'];
    const record = {};
    for (const key of allowed) {
        if (body[key] !== undefined) record[key] = body[key];
    }
    if (record.age !== undefined) {
        const n = Number(record.age);
        if (!Number.isFinite(n)) {
            return { error: 'Age must be a number.' };
        }
        record.age = n;
    }
    if (typeof record.patient === 'string') record.patient = record.patient.trim();
    if (typeof record.doctor === 'string')  record.doctor  = record.doctor.trim();
    return { record };
}

// List all patient records.
app.get('/api/records', requireAuth, async (_req, res, next) => {
    try {
        const records = await db.getAllRecords();
        res.json({ records });
    } catch (err) {
        next(err);
    }
});

// Create a new patient record from a finalized scan.
app.post('/api/records', requireAuth, async (req, res, next) => {
    try {
        const { record, error } = normalizeRecordPayload(req.body);
        if (error) return res.status(400).json({ error });
        if (!record.patient) return res.status(400).json({ error: 'Patient name is required.' });

        const result = record.result || 'Healthy';
        const severity = record.severity || 'Healthy';
        const newRecord = {
            id: req.body.id || await nextRecordId(result, severity),
            patient: record.patient,
            age: record.age ?? 0,
            gender: record.gender || 'Other',
            date: record.date || new Date().toISOString().slice(0, 10),
            result,
            confidence: record.confidence || '0%',
            doctor: record.doctor || req.user.name || 'Unassigned',
            severity,
            fundus_image: record.fundus_image || null,
            features: record.features || null,
            created_by: req.user.username,
            created_at: new Date().toISOString()
        };

        const inserted = await db.insertRecord(newRecord);
        res.status(201).json({ record: inserted });
    } catch (err) {
        next(err);
    }
});

// Update an existing patient record. The client must include the user's
// password in the body so the server can re-verify before the change.
app.patch('/api/records/:id', requireAuth, async (req, res, next) => {
    try {
        const { password, ...changes } = req.body || {};
        if (!password || password !== req.user.password) {
            return res.status(401).json({ error: 'Password verification failed.' });
        }

        const { record, error } = normalizeRecordPayload(changes);
        if (error) return res.status(400).json({ error });

        const existing = await db.findRecordById(req.params.id);
        if (!existing) return res.status(404).json({ error: 'Record not found.' });

        const patch = {
            ...record,
            updated_by: req.user.username,
            updated_at: new Date().toISOString()
        };

        const updated = await db.patchRecord(req.params.id, patch);
        res.json({ record: updated });
    } catch (err) {
        next(err);
    }
});

// Delete a patient record. Same password re-verification rules as PATCH.
app.delete('/api/records/:id', requireAuth, async (req, res, next) => {
    try {
        const password = req.body?.password || req.headers['x-reauth-password'];
        if (!password || password !== req.user.password) {
            return res.status(401).json({ error: 'Password verification failed.' });
        }

        const removed = await db.deleteRecord(req.params.id);
        if (!removed) return res.status(404).json({ error: 'Record not found.' });
        res.json({ ok: true, record: removed });
    } catch (err) {
        next(err);
    }
});

// Centralised error handler so async failures show as JSON instead of HTML.
app.use((err, _req, res, _next) => {
    console.error('[server] error', err);
    res.status(500).json({ error: err.message || 'Internal server error.' });
});

db.init().then(() => {
    app.listen(PORT, () => {
        console.log(`OcuVision Server is running on port ${PORT}`);
    });
}).catch(err => {
    console.error('[server] failed to initialise data layer:', err.message);
    process.exit(1);
});