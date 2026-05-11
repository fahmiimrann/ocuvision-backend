const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
require('dotenv').config();

const db = require('./db');
const { runMatlabModel, isMatlabModelFile } = require('./matlab-runner');

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
        fs.writeFileSync(MODEL_REGISTRY_FILE, JSON.stringify([DEFAULT_MODEL], null, 2));
    }
}

function readModels() {
    ensureModelRegistry();
    try {
        const models = JSON.parse(fs.readFileSync(MODEL_REGISTRY_FILE, 'utf8'));
        return Array.isArray(models) && models.length ? models : [DEFAULT_MODEL];
    } catch (err) {
        return [DEFAULT_MODEL];
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

// Base route to verify the API is running
app.get('/', (req, res) => {
    res.json({ 
        status: "Online", 
        project: "OcuVision AI API",
        message: "Welcome to the Ophthalmic Intelligence Backend",
        activeModel: getActiveModel().name
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
        result: 'No urgent retinal abnormality detected',
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
// If the active model is a MATLAB script (.m / .mlx) it is executed via the
// MATLAB runner; otherwise the demo backend logic returns a sample result.
app.post('/api/analyze', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded.' });
    }

    const activeModel = getActiveModel();
    const modelName = activeModel?.name || 'Demo Retina Core';
    const storedFile = activeModel?.storedFile
        ? path.join(MODEL_DIR, activeModel.storedFile)
        : null;

    let tempImagePath = null;
    try {
        // MATLAB path: only when a real .m / .mlx file has been uploaded.
        if (
            storedFile &&
            fs.existsSync(storedFile) &&
            isMatlabModelFile(activeModel.originalName || storedFile)
        ) {
            tempImagePath = writeTempImage(req.file.buffer, req.file.originalname);
            const matlabRaw = await runMatlabModel({
                scriptPath: storedFile,
                imagePath: tempImagePath
            });

            const diagnosis = matlabRaw.diagnosis || matlabRaw.result || 'No urgent retinal abnormality detected';
            const scoreNum  = Number(matlabRaw.score ?? matlabRaw.confidence ?? 0);
            const scoreStr  = Number.isFinite(scoreNum) ? scoreNum.toFixed(1) : '0.0';
            const severity  = matlabRaw.severity || 'Healthy';
            const notes     = matlabRaw.notes || 'MATLAB model analysis completed.';
            const status    = matlabRaw.status || deriveStatus(severity, diagnosis);
            const features  = matlabRaw.features && typeof matlabRaw.features === 'object'
                ? matlabRaw.features
                : null;

            return res.json({
                status,
                score: scoreStr,
                result: diagnosis,
                severity,
                notes,
                confidence: `${scoreStr}%`,
                fileName: req.file.originalname,
                features,
                model: { id: activeModel.id, name: modelName, runtime: 'matlab' }
            });
        }

        // Fallback: demo logic for the built-in model or non-MATLAB uploads.
        const result = pickDemoResult();
        return res.json({
            ...result,
            confidence: `${result.score}%`,
            fileName: req.file.originalname,
            features: result.features || null,
            model: { id: activeModel.id, name: modelName, runtime: activeModel.type || 'demo' }
        });
    } catch (err) {
        console.error('[analyze] MATLAB pipeline failed:', err.message);
        return res.status(502).json({
            error: `Active AI model failed to run: ${err.message}`,
            model: { id: activeModel.id, name: modelName }
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
function nextRecordId() {
    return `OCU-${Date.now().toString().slice(-6)}`;
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

        const newRecord = {
            id: req.body.id || nextRecordId(),
            patient: record.patient,
            age: record.age ?? 0,
            gender: record.gender || 'Other',
            date: record.date || new Date().toISOString().slice(0, 10),
            result: record.result || 'No urgent retinal abnormality detected',
            confidence: record.confidence || '0%',
            doctor: record.doctor || req.user.name || 'Unassigned',
            severity: record.severity || 'Healthy',
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
